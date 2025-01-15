from flask import Flask, render_template, jsonify

app = Flask(__name__)

# API for logs
@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = [
        {"timestamp": "2025-01-15 10:00", "message": "System started", "status": "Info"},
        {"timestamp": "2025-01-15 10:05", "message": "User login", "status": "Success"},
        {"timestamp": "2025-01-15 10:10", "message": "CPU usage high", "status": "Warning"}
    ]
    return jsonify(logs)

# API for phishing detection rate
@app.route('/api/phishing_rate', methods=['GET'])
def get_phishing_rate():
    phishing_data = {
        "total_emails": 1000,
        "phishing_emails": 120,
        "detection_rate": "12%"
    }
    return jsonify(phishing_data)

# API for threat levels
@app.route('/api/threat_levels', methods=['GET'])
def get_threat_levels():
    threat_levels = {
        "low": 80,
        "medium": 30,
        "high": 10
    }
    return jsonify(threat_levels)

# API for recent phishing alerts
@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    alerts = [
        {"timestamp": "2025-01-15 11:00", "sender": "noreply@phish.com", "subject": "Urgent: Verify Your Account", "severity": "High"},
        {"timestamp": "2025-01-15 11:15", "sender": "support@secure.com", "subject": "Password Reset Required", "severity": "Medium"},
        {"timestamp": "2025-01-15 11:30", "sender": "alerts@bank.com", "subject": "Unusual Login Activity", "severity": "Low"}
    ]
    return jsonify(alerts)

# Dashboard route
@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
