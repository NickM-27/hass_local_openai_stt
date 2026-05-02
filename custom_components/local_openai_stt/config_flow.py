"""Config flow for the Local OpenAI STT integration."""

from __future__ import annotations

import logging
from typing import Any

from openai import AsyncOpenAI, OpenAIError
import voluptuous as vol

from homeassistant.config_entries import (
    ConfigEntry,
    ConfigFlow,
    ConfigFlowResult,
    OptionsFlow,
)
from homeassistant.core import callback
from homeassistant.helpers.selector import (
    NumberSelector,
    NumberSelectorConfig,
    NumberSelectorMode,
    SelectSelector,
    SelectSelectorConfig,
    SelectSelectorMode,
    TextSelector,
    TextSelectorConfig,
    TextSelectorType,
)

from .const import (
    CONF_API_KEY,
    CONF_BASE_URL,
    CONF_DEBUG_LOG,
    CONF_DEBUG_LOG_KEEP,
    CONF_MODEL,
    CONF_PROMPT,
    CONF_TEMPERATURE,
    CONF_VAD_MIN_SPEECH_SECONDS,
    CONF_VAD_SILENCE_SECONDS,
    CONF_VAD_SPEECH_THRESHOLD,
    DEFAULT_API_KEY,
    DEFAULT_BASE_URL,
    DEFAULT_DEBUG_LOG,
    DEFAULT_DEBUG_LOG_KEEP,
    DEFAULT_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_MIN_SPEECH_SECONDS,
    DEFAULT_VAD_SILENCE_SECONDS,
    DEFAULT_VAD_SPEECH_THRESHOLD,
    DOMAIN,
)

_LOGGER = logging.getLogger(__name__)


async def _list_models(base_url: str, api_key: str) -> list[str]:
    """Fetch the list of model IDs from an OpenAI-compatible server."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key or DEFAULT_API_KEY)
    try:
        page = await client.models.list()
        return sorted({m.id for m in page.data})
    finally:
        await client.close()


def _model_selector(models: list[str]) -> Any:
    """Return a Selector for picking a model — dropdown if known, else free text."""
    if models:
        return SelectSelector(
            SelectSelectorConfig(
                options=models,
                custom_value=True,
                mode=SelectSelectorMode.DROPDOWN,
            )
        )
    return TextSelector(TextSelectorConfig(type=TextSelectorType.TEXT))


def _temperature_selector() -> NumberSelector:
    return NumberSelector(
        NumberSelectorConfig(
            min=0.0, max=1.0, step=0.05, mode=NumberSelectorMode.SLIDER
        )
    )


def _prompt_selector() -> TextSelector:
    return TextSelector(TextSelectorConfig(multiline=True))


class LocalOpenAISTTConfigFlow(ConfigFlow, domain=DOMAIN):
    """Initial setup flow."""

    VERSION = 1

    def __init__(self) -> None:
        """Initialise transient state shared across steps."""
        self._connection: dict[str, Any] = {}
        self._models: list[str] = []

    async def async_step_user(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 1: collect base URL and API key, then probe the server."""
        errors: dict[str, str] = {}

        if user_input is not None:
            base_url = user_input[CONF_BASE_URL].rstrip("/")
            api_key = user_input.get(CONF_API_KEY) or DEFAULT_API_KEY

            await self.async_set_unique_id(base_url)
            self._abort_if_unique_id_configured()

            try:
                self._models = await _list_models(base_url, api_key)
            except OpenAIError as err:
                _LOGGER.warning("Could not list models from %s: %s", base_url, err)
                errors["base"] = "cannot_connect"
            else:
                self._connection = {
                    CONF_BASE_URL: base_url,
                    CONF_API_KEY: api_key,
                }
                return await self.async_step_settings()

        defaults = user_input or {}
        schema = vol.Schema(
            {
                vol.Required(
                    CONF_BASE_URL,
                    default=defaults.get(CONF_BASE_URL, DEFAULT_BASE_URL),
                ): str,
                vol.Optional(
                    CONF_API_KEY,
                    default=defaults.get(CONF_API_KEY, ""),
                ): str,
            }
        )
        return self.async_show_form(step_id="user", data_schema=schema, errors=errors)

    async def async_step_settings(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Step 2: pick model, prompt, and temperature."""
        if user_input is not None:
            data = {**self._connection, **user_input}
            return self.async_create_entry(title="Local OpenAI STT", data=data)

        schema = vol.Schema(
            {
                vol.Required(CONF_MODEL): _model_selector(self._models),
                vol.Optional(CONF_PROMPT, default=DEFAULT_PROMPT): _prompt_selector(),
                vol.Optional(
                    CONF_TEMPERATURE, default=DEFAULT_TEMPERATURE
                ): _temperature_selector(),
            }
        )
        return self.async_show_form(step_id="settings", data_schema=schema)

    @staticmethod
    @callback
    def async_get_options_flow(config_entry: ConfigEntry) -> OptionsFlow:
        """Return the options flow handler."""
        return LocalOpenAISTTOptionsFlow()


class LocalOpenAISTTOptionsFlow(OptionsFlow):
    """Allow the user to retune model/prompt/temperature and VAD parameters."""

    async def async_step_init(
        self, user_input: dict[str, Any] | None = None
    ) -> ConfigFlowResult:
        """Single options screen."""
        if user_input is not None:
            return self.async_create_entry(title="", data=user_input)

        try:
            models = await _list_models(
                self.config_entry.data[CONF_BASE_URL],
                self.config_entry.data.get(CONF_API_KEY, DEFAULT_API_KEY),
            )
        except OpenAIError:
            models = []

        cur = {**self.config_entry.data, **self.config_entry.options}

        schema = vol.Schema(
            {
                vol.Required(
                    CONF_MODEL,
                    default=cur.get(CONF_MODEL, ""),
                ): _model_selector(models),
                vol.Optional(
                    CONF_PROMPT,
                    default=cur.get(CONF_PROMPT, DEFAULT_PROMPT),
                ): _prompt_selector(),
                vol.Optional(
                    CONF_TEMPERATURE,
                    default=cur.get(CONF_TEMPERATURE, DEFAULT_TEMPERATURE),
                ): _temperature_selector(),
                vol.Optional(
                    CONF_VAD_SILENCE_SECONDS,
                    default=cur.get(
                        CONF_VAD_SILENCE_SECONDS, DEFAULT_VAD_SILENCE_SECONDS
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.2,
                        max=3.0,
                        step=0.1,
                        mode=NumberSelectorMode.BOX,
                        unit_of_measurement="s",
                    )
                ),
                vol.Optional(
                    CONF_VAD_MIN_SPEECH_SECONDS,
                    default=cur.get(
                        CONF_VAD_MIN_SPEECH_SECONDS,
                        DEFAULT_VAD_MIN_SPEECH_SECONDS,
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        mode=NumberSelectorMode.BOX,
                        unit_of_measurement="s",
                    )
                ),
                vol.Optional(
                    CONF_VAD_SPEECH_THRESHOLD,
                    default=cur.get(
                        CONF_VAD_SPEECH_THRESHOLD, DEFAULT_VAD_SPEECH_THRESHOLD
                    ),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=0.1,
                        max=0.95,
                        step=0.05,
                        mode=NumberSelectorMode.SLIDER,
                    )
                ),
                vol.Optional(
                    CONF_DEBUG_LOG,
                    default=cur.get(CONF_DEBUG_LOG, DEFAULT_DEBUG_LOG),
                ): bool,
                vol.Optional(
                    CONF_DEBUG_LOG_KEEP,
                    default=cur.get(CONF_DEBUG_LOG_KEEP, DEFAULT_DEBUG_LOG_KEEP),
                ): NumberSelector(
                    NumberSelectorConfig(
                        min=1,
                        max=200,
                        step=1,
                        mode=NumberSelectorMode.BOX,
                    )
                ),
            }
        )
        return self.async_show_form(step_id="init", data_schema=schema)
