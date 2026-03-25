"""
Earthquake Prediction API
Simple API to predict next earthquake

Usage:
    from api import EarthquakePredictor

    predictor = EarthquakePredictor()
    result = predictor.predict(
        region_code="R221_570",
        recent_events=[...]  # optional
    )

Author: haind
Date: 2025-03-25
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
from data_preparer import DataPreparer
from model_builder import EarthquakeLSTM


class EarthquakePredictor:
    """
    Main API for earthquake prediction

    This class provides a simple interface for:
    - Training models
    - Making predictions
    - Loading/saving models
    """

    def __init__(self, model_path=None):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model (optional)
        """
        self.model_path = model_path
        self.preparer = DataPreparer()
        self.model_builder = None
        self.region_encoder = None

        # Load data
        print("Loading earthquake data...")
        self.preparer.load_data()

        # Get available regions
        self.available_regions = self.preparer.get_regions()

        print(f"✓ Loaded {len(self.preparer.data)} events")
        print(f"✓ Available regions: {len(self.available_regions)}")

        # Load model if provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train_model(self, region_code=None, min_events=100, epochs=50):
        """
        Train LSTM model

        Args:
            region_code: Specific region or None for all regions
            min_events: Minimum events required per region
            epochs: Number of training epochs

        Returns:
            Training history
        """
        if region_code:
            # Train on specific region
            region_data = self.preparer.filter_by_region(region_code, min_events)
            X, y = self.preparer.prepare_sequences(region_data)

            # Split and scale
            (X_train, y_train), (X_val, y_val), _ = self.preparer.split_data(X, y)
            scaled = self.preparer.scale_features(X_train, X_val)

            # Build and train
            n_features = X_train.shape[-1]
            self.model_builder = EarthquakeLSTM(n_features=n_features)
            self.model_builder.build_model()

            history = self.model_builder.train(
                scaled['X_train'], y_train,
                scaled['X_val'], y_val,
                epochs=epochs
            )

            # Save model
            model_path = MODEL_DIR / f'model_{region_code}.keras'
            self.model_builder.save_model(model_path)

            return history

        else:
            # Train on all regions
            raise NotImplementedError("Training on all regions not yet implemented")

    def load_model(self, model_path):
        """
        Load trained model

        Args:
            model_path: Path to model file
        """
        self.model_path = model_path

        # Get number of features from config
        n_features = len(INPUT_FEATURES) - 1  # Exclude 'time'

        self.model_builder = EarthquakeLSTM(n_features=n_features)
        self.model_builder.load_model(model_path)

        print(f"✓ Model loaded: {model_path}")

    def predict(self, region_code, recent_events=None):
        """
        Predict next earthquake

        Args:
            region_code: Region identifier (e.g., 'R221_570')
            recent_events: Optional list of recent events (user input)
                           If None, uses only historical data

        Returns:
            Dictionary with prediction results
        """
        print(f"\nPredicting for region: {region_code}")

        # Get historical data for region
        try:
            historical_data = self.preparer.filter_by_region(
                region_code,
                min_events=SEQUENCE_LENGTH
            )
        except ValueError as e:
            return {
                'error': str(e),
                'message': f'Not enough data for region {region_code}'
            }

        # Add user input if provided
        if recent_events:
            df_input = pd.DataFrame(recent_events)
            df_input['time'] = pd.to_datetime(df_input['time'])

            # Combine
            all_data = pd.concat([historical_data, df_input], ignore_index=True)
            all_data = all_data.sort_values('time').reset_index(drop=True)

            print(f"  Historical: {len(historical_data)}, User input: {len(df_input)}")
        else:
            all_data = historical_data
            print(f"  Historical: {len(historical_data)}")

        # Prepare input
        try:
            X = self.preparer.prepare_for_prediction(all_data)
        except ValueError as e:
            return {
                'error': str(e),
                'message': 'Need more events for prediction'
            }

        # Load model if not loaded
        if self.model_builder is None:
            # Try region-specific model first
            model_path = MODEL_DIR / f'model_{region_code}.keras'
            if not os.path.exists(model_path):
                model_path = MODEL_DIR / 'model_all_regions.keras'

            if not os.path.exists(model_path):
                return {
                    'error': 'Model not found',
                    'message': 'Please train model first',
                    'suggestion': f'Run: python train.py --region {region_code}'
                }

            self.load_model(model_path)

        # Make prediction
        prediction = self.model_builder.predict(X)

        # Calculate time
        last_event_time = all_data['time'].iloc[-1]
        time_to_next = prediction['time_to_next']
        next_event_time = last_event_time + timedelta(seconds=time_to_next)

        # Prepare result
        result = {
            'region_code': region_code,
            'last_event': {
                'time': str(last_event_time),
                'magnitude': float(all_data['mag'].iloc[-1]),
                'latitude': float(all_data['latitude'].iloc[-1]),
                'longitude': float(all_data['longitude'].iloc[-1])
            },
            'prediction': {
                'time': str(next_event_time),
                'time_to_next_seconds': float(time_to_next),
                'time_to_next_hours': float(time_to_next / 3600),
                'magnitude': float(prediction['next_mag']),
                'is_m5_plus': bool(prediction['next_mag_binary'] >= 0.5),
                'confidence': float(max(prediction['next_mag_binary'], 1 - prediction['next_mag_binary']))
            },
            'risk_assessment': self._assess_risk(prediction)
        }

        return result

    def _assess_risk(self, prediction):
        """Assess risk level"""
        mag = prediction['next_mag']
        is_m5 = prediction['next_mag_binary'] >= 0.5

        if mag >= 7.0:
            return {
                'level': 'CRITICAL',
                'color': 'red',
                'message': f'M{mag:.1f} earthquake predicted - EXTREME danger'
            }
        elif mag >= 6.0:
            return {
                'level': 'HIGH',
                'color': 'orange',
                'message': f'M{mag:.1f} earthquake predicted - HIGH danger'
            }
        elif mag >= 5.0:
            return {
                'level': 'MODERATE',
                'color': 'yellow',
                'message': f'M{mag:.1f} earthquake predicted - MODERATE danger'
            }
        else:
            return {
                'level': 'LOW',
                'color': 'green',
                'message': f'M{mag:.1f} earthquake predicted - Low danger'
            }

    def get_region_info(self, region_code):
        """
        Get information about a region

        Args:
            region_code: Region identifier

        Returns:
            Dictionary with region statistics
        """
        region_data = self.preparer.data[
            self.preparer.data['region_code'] == region_code
        ]

        if len(region_data) == 0:
            return {'error': f'Region {region_code} not found'}

        return {
            'region_code': region_code,
            'total_events': len(region_data),
            'date_range': {
                'start': str(region_data['time'].min()),
                'end': str(region_data['time'].max())
            },
            'magnitude_stats': {
                'min': float(region_data['mag'].min()),
                'max': float(region_data['mag'].max()),
                'mean': float(region_data['mag'].mean()),
                'std': float(region_data['mag'].std())
            },
            'last_events': [
                {
                    'time': str(row['time']),
                    'mag': float(row['mag']),
                    'lat': float(row['latitude']),
                    'lon': float(row['longitude'])
                }
                for _, row in region_data.tail(5).iterrows()
            ]
        }

    def list_regions(self, min_events=10, limit=20):
        """
        List available regions

        Args:
            min_events: Minimum events required
            limit: Maximum number of regions to return

        Returns:
            List of regions with statistics
        """
        region_stats = []

        for region in self.available_regions:
            region_data = self.preparer.data[
                self.preparer.data['region_code'] == region
            ]

            if len(region_data) >= min_events:
                region_stats.append({
                    'region_code': region,
                    'event_count': len(region_data),
                    'max_mag': float(region_data['mag'].max()),
                    'last_event': str(region_data['time'].max())
                })

        # Sort by event count (descending) and limit
        region_stats.sort(key=lambda x: x['event_count'], reverse=True)

        return region_stats[:limit]


# Example usage
if __name__ == "__main__":
    print("="*70)
    print(" EARTHQUAKE PREDICTION API TEST")
    print("="*70)

    # Initialize predictor
    predictor = EarthquakePredictor()

    # List available regions
    print("\n1. Available regions:")
    regions = predictor.list_regions(min_events=50, limit=10)
    for r in regions:
        print(f"  {r['region_code']}: {r['event_count']} events, M{r['max_mag']:.1f} max")

    # Test with first region
    if len(regions) > 0:
        test_region = regions[0]['region_code']
        print(f"\n2. Testing prediction for {test_region}")
        result = predictor.predict(test_region)

        if 'error' not in result:
            print(f"\n  Last event: {result['last_event']['time']}")
            print(f"  Predicted: {result['prediction']['time']}")
            print(f"  Magnitude: M{result['prediction']['magnitude']:.1f}")
            print(f"  Risk: {result['risk_assessment']['level']}")

    print("\n" + "="*70)
    print(" TEST COMPLETED")
    print("="*70)
