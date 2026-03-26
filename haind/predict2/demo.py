"""
Flask Demo App for Earthquake Prediction
Select seismic zone and show prediction results on UI
"""

import os
import sys
import json
import shutil
from flask import Flask, render_template, jsonify, request
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import SEQUENCE_LENGTH
from predict import main as predict_main, load_models, get_device, prepare_features

app = Flask(__name__)

# 5 Seismic Zones for demo
SEISMIC_ZONES = [
    {
        'id': 'japan',
        'name': 'Japan',
        'location': 'Tohoku Region, Honshu - East Coast',
        'mainshock': 'M5.9',
        'description': 'Honshu Region - 2011 Tohoku Earthquake Sequence',
        'color': '#ef4444',
        'file': 'input_events_japan.json'
    },
    {
        'id': 'philippines',
        'name': 'Philippines',
        'location': 'Mindanao Region - Southern Philippines',
        'mainshock': 'M5.5',
        'description': 'Mindanao Region - Active Seismic Zone',
        'color': '#f59e0b',
        'file': 'input_events_philippines.json'
    },
    {
        'id': 'chile',
        'name': 'Chile',
        'location': 'Central Chile - Near Santiago',
        'mainshock': 'M5.3',
        'description': 'Central Chile - Subduction Zone',
        'color': '#10b981',
        'file': 'input_events_chile.json'
    },
    {
        'id': 'indonesia',
        'name': 'Indonesia - Sumatra',
        'location': 'Northern Sumatra - Aceh Province',
        'mainshock': 'M5.0',
        'description': 'Sumatra Region - Sunda Arc',
        'color': '#3b82f6',
        'file': 'input_events_indonesia__sumatra.json'
    },
    {
        'id': 'california',
        'name': 'USA - California',
        'location': 'Baja California - Mexicali Area',
        'mainshock': 'M5.3',
        'description': 'Baja California - San Andreas Fault',
        'color': '#8b5cf6',
        'file': 'input_events_usa__california.json'
    }
]

INPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(INPUT_DIR, 'prediction_results.json')


def run_prediction(zone_id):
    """Run prediction for selected zone"""
    zone = next(z for z in SEISMIC_ZONES if z['id'] == zone_id)

    # Copy selected input file to input_events.json
    src_file = os.path.join(INPUT_DIR, zone['file'])
    dst_file = os.path.join(INPUT_DIR, 'input_events.json')

    if not os.path.exists(src_file):
        return {'error': f'Input file not found: {zone["file"]}'}

    shutil.copy(src_file, dst_file)

    # Run prediction (capture output)
    import io
    import contextlib

    # Redirect stdout to capture print output
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
        try:
            predict_main()
        except Exception as e:
            return {'error': str(e)}

    # Read results
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r') as f:
            results = json.load(f)
        return results
    else:
        return {'error': 'Prediction failed - no output file'}


@app.route('/')
def index():
    """Render demo page"""
    return render_template('demo.html', zones=SEISMIC_ZONES)


@app.route('/api/predict', methods=['POST'])
def predict():
    """Run prediction for selected zone"""
    data = request.json
    zone_id = data.get('zone_id')

    if not zone_id:
        return jsonify({'error': 'Missing zone_id'}), 400

    # Check if zone exists
    if zone_id not in [z['id'] for z in SEISMIC_ZONES]:
        return jsonify({'error': 'Invalid zone_id'}), 400

    # Run prediction
    result = run_prediction(zone_id)

    if 'error' in result:
        return jsonify(result), 500

    return jsonify(result)


@app.route('/api/zones')
def get_zones():
    """Get list of seismic zones"""
    return jsonify(SEISMIC_ZONES)


@app.route('/api/events/<zone_id>')
def get_events(zone_id):
    """Get events for a specific zone (use simulation file if exists)"""
    zone = next((z for z in SEISMIC_ZONES if z['id'] == zone_id), None)
    if not zone:
        return jsonify({'error': 'Zone not found'}), 404

    # Check if simulation file exists (zone-specific)
    sim_file = os.path.join(INPUT_DIR, f'simulation_{zone_id}.json')
    if os.path.exists(sim_file):
        events_file = sim_file
        is_simulation = True
    else:
        events_file = os.path.join(INPUT_DIR, zone['file'])
        is_simulation = False
        if not os.path.exists(events_file):
            return jsonify({'error': 'Input file not found'}), 404

    with open(events_file, 'r') as f:
        events = json.load(f)

    # Get ground truth magnitude from last event (only for original file)
    gt_mag = None
    if not is_simulation and events:
        gt_mag = events[-1].get('target_next_mag', None)

    # Format events for display
    formatted_events = []
    for i, event in enumerate(events):
        formatted_events.append({
            'index': i + 1,
            'time': event.get('time', ''),
            'mag': event.get('mag', 0),
            'depth': event.get('depth', 0),
            'latitude': event.get('latitude', 0),
            'longitude': event.get('longitude', 0)
        })

    return jsonify({
        'zone': zone,
        'is_simulation': is_simulation,
        'total_events': len(events),
        'ground_truth_mag': gt_mag,
        'events': formatted_events
    })


@app.route('/api/simulate/start', methods=['POST'])
def start_simulation():
    """Start real-time simulation for a zone"""
    data = request.json
    zone_id = data.get('zone_id')

    if not zone_id:
        return jsonify({'error': 'Missing zone_id'}), 400

    zone = next((z for z in SEISMIC_ZONES if z['id'] == zone_id), None)
    if not zone:
        return jsonify({'error': 'Zone not found'}), 404

    # Create simulation file from original input
    input_file = os.path.join(INPUT_DIR, zone['file'])
    sim_file = os.path.join(INPUT_DIR, f'simulation_{zone_id}.json')

    if not os.path.exists(input_file):
        return jsonify({'error': 'Input file not found'}), 404

    # Copy original events to simulation file
    with open(input_file, 'r') as f:
        all_events = json.load(f)

    # Use only first SEQUENCE_LENGTH events for simulation (will append one by one)
    sim_events = all_events[:SEQUENCE_LENGTH] if len(all_events) >= SEQUENCE_LENGTH else all_events

    with open(sim_file, 'w') as f:
        json.dump(sim_events, f, indent=2)

    return jsonify({
        'success': True,
        'total_available': len(all_events),
        'initial_events': len(sim_events),
        'remaining': len(all_events) - len(sim_events)
    })


@app.route('/api/simulate/next', methods=['POST'])
def simulate_next_event():
    """Simulate adding next event and run prediction"""
    data = request.json
    zone_id = data.get('zone_id')

    if not zone_id:
        return jsonify({'error': 'Missing zone_id'}), 400

    zone = next((z for z in SEISMIC_ZONES if z['id'] == zone_id), None)
    if not zone:
        return jsonify({'error': 'Zone not found'}), 404

    # Files
    input_file = os.path.join(INPUT_DIR, zone['file'])
    sim_file = os.path.join(INPUT_DIR, f'simulation_{zone_id}.json')
    target_file = os.path.join(INPUT_DIR, 'input_events.json')

    # Read all original events
    with open(input_file, 'r') as f:
        all_events = json.load(f)

    # Read current simulation events
    if os.path.exists(sim_file):
        with open(sim_file, 'r') as f:
            sim_events = json.load(f)
    else:
        sim_events = []

    # Check if there are more events to add
    current_index = len(sim_events)
    if current_index >= len(all_events):
        return jsonify({
            'success': True,
            'finished': True,
            'message': 'Simulation completed - all events processed'
        })

    # Add next event
    next_event = all_events[current_index]
    sim_events.append(next_event)

    # Save updated simulation
    with open(sim_file, 'w') as f:
        json.dump(sim_events, f, indent=2)

    # Copy to input_events.json for prediction
    with open(target_file, 'w') as f:
        json.dump(sim_events, f, indent=2)

    # Calculate Ground Truth: collect ALL M5+ within 7 days after appended event
    from datetime import datetime, timedelta

    ground_truth_list = []
    appended_time = datetime.fromisoformat(next_event['time'].replace('Z', '+00:00'))
    seven_days_later = appended_time + timedelta(days=7)

    # Check all events in original data after the current index
    for future_event in all_events[current_index + 1:]:
        event_time = datetime.fromisoformat(future_event['time'].replace('Z', '+00:00'))
        if event_time > seven_days_later:
            break  # Beyond 7 days, stop checking
        if future_event.get('mag', 0) >= 5.0:
            # Calculate detailed time difference
            time_diff = event_time - appended_time
            total_seconds = int(time_diff.total_seconds())
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60

            ground_truth_list.append({
                'mag': future_event['mag'],
                'time': future_event['time'],
                'days_after': days,
                'hours': hours,
                'minutes': minutes
            })

    ground_truth = ground_truth_list if ground_truth_list else None

    # Run prediction
    try:
        import io
        import contextlib

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            predict_main()

        # Read results
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r') as f:
                results = json.load(f)

            return jsonify({
                'success': True,
                'finished': False,
                'event_added': {
                    'index': current_index + 1,
                    'mag': next_event.get('mag', 0),
                    'time': next_event.get('time', ''),
                    'depth': next_event.get('depth', 0)
                },
                'total_events': len(sim_events),
                'ground_truth': ground_truth,  # None if no M5+ within 7 days
                'prediction': results
            })
        else:
            return jsonify({
                'success': True,
                'finished': False,
                'event_added': {
                    'index': current_index + 1,
                    'mag': next_event.get('mag', 0),
                    'time': next_event.get('time', '')
                },
                'total_events': len(sim_events),
                'ground_truth': ground_truth,
                'error': 'Prediction failed'
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/simulate/reset', methods=['POST'])
def reset_simulation():
    """Reset simulation for a zone"""
    data = request.json
    zone_id = data.get('zone_id')

    if zone_id:
        sim_file = os.path.join(INPUT_DIR, f'simulation_{zone_id}.json')
    else:
        # If no zone_id specified, remove all simulation files
        import glob
        sim_files = glob.glob(os.path.join(INPUT_DIR, 'simulation_*.json'))
        for sim_file in sim_files:
            os.remove(sim_file)
        return jsonify({'success': True})

    if os.path.exists(sim_file):
        os.remove(sim_file)

    return jsonify({'success': True})


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" EARTHQUAKE PREDICTION DEMO ".center(60))
    print("="*60)
    print("\nStarting Flask server...")
    print("Open browser: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=5000)


if __name__ == '__main__':
    print("\n" + "="*60)
    print(" EARTHQUAKE PREDICTION DEMO ".center(60))
    print("="*60)
    print("\nStarting Flask server...")
    print("Open browser: http://localhost:5000")
    print("\nPress Ctrl+C to stop\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
