import asyncio
from typing import Optional

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("weather-minimal")

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


async def _geocode(city: str) -> Optional[dict]:
    params = {"name": city, "count": 1}
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.get(GEOCODE_URL, params=params)
        r.raise_for_status()
        data = r.json() or {}
    results = data.get("results") or []
    return results[0] if results else None


def _fmt_coord(lat: float, lon: float) -> str:
    def d(v: float, lataxis: bool) -> str:
        hemi = ("N" if v >= 0 else "S") if lataxis else ("E" if v >= 0 else "W")
        return f"{abs(v):.3f}°{hemi}"
    return f"{d(lat, True)}, {d(lon, False)}"


# -------------------- TOOLS --------------------

@mcp.tool()
async def get_current_weather(city: str) -> str:
    """
    Current weather by city name (global).
    Args:
        city: e.g., "Paris", "New York", "Bengaluru"
    Returns:
        Human-friendly lines of text.
    """
    hit = await _geocode(city)
    if not hit:
        return f"Could not geocode '{city}'. Try a more specific name."

    lat, lon = float(hit["latitude"]), float(hit["longitude"])
    q = {
        "latitude": lat,
        "longitude": lon,
        "current_weather": "true",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(FORECAST_URL, params=q)
            r.raise_for_status()
            data = r.json() or {}
    except httpx.HTTPError as e:
        return f"Network error contacting weather API: {e}"

    cur = data.get("current_weather") or {}
    temp = cur.get("temperature")
    wind = cur.get("windspeed")
    code = cur.get("weathercode")
    loc = f"{hit.get('name')}, {hit.get('country_code')} ({_fmt_coord(lat, lon)})"
    tz = data.get("timezone", "local")

    return (
        f"Current weather for {loc} [{tz}]\n"
        f"- Temperature: {temp} °C\n"
        f"- Wind: {wind} km/h\n"
        f"- Weather code: {code} (WMO)\n"
        f"- Source: Open-Meteo"
    )


@mcp.tool()
async def get_daily_forecast(latitude: float, longitude: float, days: int = 3) -> str:
    """
    Compact daily forecast by coordinates.
    Args:
        latitude: decimal degrees (+N / -S)
        longitude: decimal degrees (+E / -W)
        days: 1-7
    Returns:
        Human-friendly lines of text.
    """
    days = max(1, min(7, int(days)))
    q = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "timezone": "auto",
    }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.get(FORECAST_URL, params=q)
            r.raise_for_status()
            data = r.json() or {}
    except httpx.HTTPError as e:
        return f"Network error contacting weather API: {e}"

    daily = data.get("daily") or {}
    dates = daily.get("time") or []
    tmax = daily.get("temperature_2m_max") or []
    tmin = daily.get("temperature_2m_min") or []
    precip = daily.get("precipitation_sum") or []
    wcode = daily.get("weathercode") or []
    tz = data.get("timezone", "local")

    if not dates:
        return "No forecast data available."

    n = min(days, len(dates))
    lines = [
        f"Daily forecast for {_fmt_coord(latitude, longitude)} [{tz}] (next {n} day(s))"
    ]
    for i in range(n):
        lines.append(
            f"- {dates[i]}: {tmin[i]:.1f}–{tmax[i]:.1f} °C, precip {precip[i]:.1f} mm, code {wcode[i]}"
        )
    lines.append("Source: Open-Meteo")
    return "\n".join(lines)


def main() -> None:
    # Use stdio transport for Claude Desktop / MCP CLI / Cursor
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
