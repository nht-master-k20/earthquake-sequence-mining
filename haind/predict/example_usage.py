"""
Example Usage Script
Demonstrates how to use the Earthquake Prediction API

Author: haind
Date: 2025-03-25
"""

import json
from api import EarthquakePredictor


def example_1_list_regions():
    """Example 1: List available regions"""
    print("\n" + "="*70)
    print(" EXAMPLE 1: LIST AVAILABLE REGIONS")
    print("="*70)

    predictor = EarthquakePredictor()

    # List regions with at least 50 events
    regions = predictor.list_regions(min_events=50, limit=10)

    print(f"\nFound {len(regions)} regions with >= 50 events:\n")
    for i, r in enumerate(regions):
        print(f"  {i+1}. {r['region_code']}")
        print(f"     Events: {r['event_count']}")
        print(f"     Max Mag: M{r['max_mag']:.1f}")
        print(f"     Last: {r['last_event']}")


def example_2_predict_from_historical():
    """Example 2: Predict using only historical data"""
    print("\n" + "="*70)
    print(" EXAMPLE 2: PREDICT FROM HISTORICAL DATA")
    print("="*70)

    predictor = EarthquakePredictor()

    # Get first available region with enough data
    regions = predictor.list_regions(min_events=SEQUENCE_LENGTH, limit=1)

    if regions:
        region_code = regions[0]['region_code']

        # Get region info first
        info = predictor.get_region_info(region_code)
        print(f"\nRegion Info: {region_code}")
        print(f"  Total events: {info['total_events']}")
        print(f"  Magnitude range: M{info['magnitude_stats']['min']:.1f} - M{info['magnitude_stats']['max']:.1f}")

        # Predict
        print(f"\nMaking prediction...")
        result = predictor.predict(region_code)

        if 'error' not in result:
            print(f"\n✓ Prediction successful!")
            print(f"\nLast event:")
            print(f"  Time: {result['last_event']['time']}")
            print(f"  Mag:  M{result['last_event']['magnitude']:.1f}")

            print(f"\nPredicted next event:")
            print(f"  Time: {result['prediction']['time']}")
            print(f"  In: {result['prediction']['time_to_next_hours']:.1f} hours")
            print(f"  Mag: M{result['prediction']['magnitude']:.1f}")
            print(f"  M5+: {result['prediction']['is_m5_plus']}")

            print(f"\nRisk Assessment:")
            print(f"  Level: {result['risk_assessment']['level']}")
            print(f"  {result['risk_assessment']['message']}")


def example_3_predict_from_user_input():
    """Example 3: Predict using user input + historical data"""
    print("\n" + "="*70)
    print(" EXAMPLE 3: PREDICT FROM USER INPUT")
    print("="*70)

    predictor = EarthquakePredictor()

    # Load example input
    with open('example_input.json', 'r') as f:
        user_events = json.load(f)

    print(f"\nUser provided {len(user_events)} events:")
    for i, event in enumerate(user_events):
        print(f"  {i+1}. {event['time']} - M{event['mag']:.1f}")

    # Predict (will automatically determine region from first event)
    print(f"\nDetermining region from first event...")
    result = predictor.predict(region_code=None, recent_events=user_events)

    if 'error' not in result:
        print(f"\n✓ Prediction successful!")
        print(f"\nLast event:")
        print(f"  Time: {result['last_event']['time']}")
        print(f"  Mag:  M{result['last_event']['magnitude']:.1f}")

        print(f"\nPredicted next event:")
        print(f"  Time: {result['prediction']['time']}")
        print(f"  In: {result['prediction']['time_to_next_hours']:.1f} hours")
        print(f"  Mag: M{result['prediction']['magnitude']:.1f}")
        print(f"  M5+: {result['prediction']['is_m5_plus']}")

        print(f"\nRisk Assessment:")
        print(f"  {result['risk_assessment']['level']}")
        print(f"  {result['risk_assessment']['message']}")


def example_4_api_usage():
    """Example 4: Using as an API in your application"""
    print("\n" + "="*70)
    print(" EXAMPLE 4: API USAGE IN YOUR APPLICATION")
    print("="*70)

    print("""
# In your application:

from predict.api import EarthquakePredictor

# Initialize (load data and model)
predictor = EarthquakePredictor(model_path='models/model_R221_570.keras')

# Real-time prediction
recent_earthquakes = [
    {
        'time': '2025-03-25 14:00:00',
        'latitude': 20.5,
        'longitude': 105.0,
        'depth': 10.5,
        'mag': 4.5,
        'sig': 250,
        'mmi': 4.2,
        'cdi': 3.5,
        'felt': 120
    },
    # ... more events
]

# Predict
result = predictor.predict(
    region_code='R221_570',
    recent_events=recent_earthquakes
)

# Use prediction
if result['prediction']['is_m5_plus']:
    send_alert(result)
else:
    log_event(result)

print(f"  Next earthquake in: {result['prediction']['time_to_next_hours']:.1f} hours")
print(f"  Magnitude: M{result['prediction']['magnitude']:.1f}")
    """)


def main():
    """Run all examples"""
    from config import SEQUENCE_LENGTH

    print("\n" + "="*70)
    print(" EARTHQUAKE PREDICTION API - EXAMPLE USAGE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sequence length: {SEQUENCE_LENGTH}")
    print(f"  Input features: 26")

    # Run examples
    example_1_list_regions()

    # Uncomment to run other examples:
    # example_2_predict_from_historical()
    # example_3_predict_from_user_input()
    # example_4_api_usage()

    print("\n" + "="*70)
    print(" EXAMPLE USAGE DEMO COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
