import requests


def get_weather(city):
    """
    Get current weather information for a city.
    No API key required.
    """
    try:
        url = f"https://wttr.in/{city}?format=j1"

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        data = response.json()
        current = data["current_condition"][0]

        weather_info = {
            "city": city,
            "temperature_c": current["temp_C"],
            "feels_like_c": current["FeelsLikeC"],
            "humidity": current["humidity"],
            "weather": current["weatherDesc"][0]["value"],
            "wind_speed_kmph": current["windspeedKmph"]
        }

        return weather_info

    except Exception as e:
        return {"error": str(e)}


# Function Call
if __name__ == "__main__":
    city = input("Enter city name: ")

    weather = get_weather(city)

    if "error" in weather:
        print(f"Error: {weather['error']}")
    else:
        print("\nCurrent Weather")
        print("-" * 30)
        print(f"City            : {weather['city']}")
        print(f"Temperature     : {weather['temperature_c']}°C")
        print(f"Feels Like      : {weather['feels_like_c']}°C")
        print(f"Humidity        : {weather['humidity']}%")
        print(f"Condition       : {weather['weather']}")
        print(f"Wind Speed      : {weather['wind_speed_kmph']} km/h")