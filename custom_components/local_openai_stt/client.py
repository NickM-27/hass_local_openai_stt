"""OpenAI client construction for the Local OpenAI STT integration."""

from __future__ import annotations

from openai import AsyncOpenAI

from homeassistant.core import HomeAssistant


def _create_client(base_url: str, api_key: str) -> AsyncOpenAI:
    """Build the client and resolve the resources this integration calls."""
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    # Newer ``openai`` releases import each resource module on first attribute
    # access (via ``importlib.import_module``), so touch the ones we use here
    # rather than on the first request.
    client.audio.transcriptions  # noqa: B018
    client.models  # noqa: B018
    return client


async def async_create_client(
    hass: HomeAssistant, base_url: str, api_key: str
) -> AsyncOpenAI:
    """Return an AsyncOpenAI client built in the executor.

    The constructor builds an httpx client, which loads the system CA bundle
    from disk, and the resource accessors may import modules. Both block, so
    neither may run on the event loop.
    """
    return await hass.async_add_executor_job(_create_client, base_url, api_key)
