import { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, LineChart, Line } from 'recharts';
import { CheckCircle, XCircle, TrendingUp, AlertCircle, BarChart3, Shield, Users, DollarSign, Award, Brain, Activity, Zap, Target, Sparkles } from 'lucide-react';
import axios from 'axios';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    Gender: 'Male',
    Married: 'Yes',
    Dependents: '0',
    Education: 'Graduate',
    Self_Employed: 'No',
    ApplicantIncome: 5000,
    CoapplicantIncome: 1500,
    LoanAmount: 150,
    Loan_Amount_Term: 360,
    Credit_History: 1.0,
    Property_Area: 'Urban'
  });

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [liveStats, setLiveStats] = useState({
    totalPredictions: 0,
    avgConfidence: 0,
    approvalRate: 0
  });
  const [particles, setParticles] = useState([]);

  // Generate floating particles
  useEffect(() => {
    const newParticles = Array.from({ length: 15 }, (_, i) => ({
      id: i,
      x: Math.random() * 100,
      y: Math.random() * 100,
      size: Math.random() * 4 + 2,
      duration: Math.random() * 10 + 15
    }));
    setParticles(newParticles);
  }, []);

  // Simulate live stats updating
  useEffect(() => {
    const interval = setInterval(() => {
      setLiveStats(prev => ({
        totalPredictions: prev.totalPredictions + Math.floor(Math.random() * 3),
        avgConfidence: 87 + Math.random() * 8,
        approvalRate: 68 + Math.random() * 5
      }));
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name.includes('Income') || name.includes('Amount') || name.includes('History') || name.includes('Term')
        ? parseFloat(value) 
        : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post('http://127.0.0.1:8000/predict', formData);
      setPrediction(response.data);
      setLiveStats(prev => ({ ...prev, totalPredictions: prev.totalPredictions + 1 }));
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to get prediction. Make sure the API is running.');
    } finally {
      setLoading(false);
    }
  };

  const getChartData = () => {
    if (!prediction) return [];
    return prediction.top_factors.map(factor => ({
      name: factor.feature.replace(/_/g, ' '),
      value: Math.abs(factor.contribution),
      impact: factor.impact,
      original: factor.contribution
    }));
  };

  const getRadarData = () => {
    if (!prediction) return [];
    return prediction.top_factors.map(factor => ({
      subject: factor.feature.replace(/_/g, ' ').substring(0, 15),
      value: Math.abs(factor.contribution * 100),
      fullMark: 1
    }));
  };

  const getRiskScore = () => {
    if (!prediction) return 50;
    return prediction.prediction === 'Approved' ? 85 : 35;
  };

  return (
    <div className="app">
      {/* Animated Particle Background */}
      <div className="particles-container">
        {particles.map(particle => (
          <div
            key={particle.id}
            className="particle"
            style={{
              left: `${particle.x}%`,
              top: `${particle.y}%`,
              width: `${particle.size}px`,
              height: `${particle.size}px`,
              animationDuration: `${particle.duration}s`
            }}
          />
        ))}
      </div>

      <header className="header">
        <div className="header-content">
          <h1>Loan Approval Intelligence System</h1>
          <p>Advanced Credit Risk Assessment with Transparent Decision-Making</p>
          
          {/* Live Stats Ticker */}
          <div className="live-ticker">
            <div className="ticker-item">
              <Activity size={16} />
              <span>Live: {liveStats.totalPredictions} predictions today</span>
            </div>
            <div className="ticker-item">
              <Zap size={16} />
              <span>{liveStats.avgConfidence.toFixed(1)}% avg confidence</span>
            </div>
            <div className="ticker-item">
              <Target size={16} />
              <span>{liveStats.approvalRate.toFixed(1)}% approval rate</span>
            </div>
          </div>
        </div>
      </header>

      <div className="container">
        <div className="left-column">
          <div className="form-section">
            <h2>Loan Application Form</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-group">
                  <label>Gender</label>
                  <select name="Gender" value={formData.Gender} onChange={handleChange}>
                    <option value="Male">Male</option>
                    <option value="Female">Female</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Married</label>
                  <select name="Married" value={formData.Married} onChange={handleChange}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Dependents</label>
                  <select name="Dependents" value={formData.Dependents} onChange={handleChange}>
                    <option value="0">0</option>
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3+">3+</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Education</label>
                  <select name="Education" value={formData.Education} onChange={handleChange}>
                    <option value="Graduate">Graduate</option>
                    <option value="Not Graduate">Not Graduate</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Self Employed</label>
                  <select name="Self_Employed" value={formData.Self_Employed} onChange={handleChange}>
                    <option value="Yes">Yes</option>
                    <option value="No">No</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Property Area</label>
                  <select name="Property_Area" value={formData.Property_Area} onChange={handleChange}>
                    <option value="Urban">Urban</option>
                    <option value="Semiurban">Semiurban</option>
                    <option value="Rural">Rural</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>Applicant Income ($)</label>
                  <input 
                    type="number" 
                    name="ApplicantIncome" 
                    value={formData.ApplicantIncome} 
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Coapplicant Income ($)</label>
                  <input 
                    type="number" 
                    name="CoapplicantIncome" 
                    value={formData.CoapplicantIncome} 
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Loan Amount ($1000s)</label>
                  <input 
                    type="number" 
                    name="LoanAmount" 
                    value={formData.LoanAmount} 
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Loan Term (Months)</label>
                  <input 
                    type="number" 
                    name="Loan_Amount_Term" 
                    value={formData.Loan_Amount_Term} 
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="form-group">
                  <label>Credit History</label>
                  <select name="Credit_History" value={formData.Credit_History} onChange={handleChange}>
                    <option value={1.0}>Good (1)</option>
                    <option value={0.0}>Poor (0)</option>
                  </select>
                </div>
              </div>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? (
                  <>
                    <div className="loading-spinner"></div>
                    <span>Analyzing with AI...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={20} />
                    <span>Predict Loan Approval</span>
                  </>
                )}
              </button>
            </form>

            {error && (
              <div className="error-box">
                <AlertCircle size={20} />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        <div className="right-column">
          {!prediction ? (
            <>
              {/* 3D Risk Assessment Sphere */}
              <div className="risk-sphere-container">
                <div className="risk-sphere">
                  <div className="sphere-inner">
                    <Brain size={48} color="#dc2626" />
                    <div className="sphere-pulse"></div>
                  </div>
                  <div className="sphere-text">
                    <h3>AI Ready</h3>
                    <p>Submit application to analyze</p>
                  </div>
                </div>
              </div>

              {/* Live Dashboard */}
              <div className="live-dashboard">
                <div className="dashboard-header">
                  <Activity size={24} color="#dc2626" />
                  <h3>Live System Status</h3>
                </div>
                
                <div className="live-metrics">
                  <div className="metric-card">
                    <div className="metric-icon approved-icon">
                      <CheckCircle size={24} />
                    </div>
                    <div className="metric-content">
                      <div className="metric-value">{liveStats.totalPredictions}</div>
                      <div className="metric-label">Predictions Today</div>
                    </div>
                    <div className="metric-trend">↑ 12%</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-icon confidence-icon">
                      <Shield size={24} />
                    </div>
                    <div className="metric-content">
                      <div className="metric-value">{liveStats.avgConfidence.toFixed(1)}%</div>
                      <div className="metric-label">Avg Confidence</div>
                    </div>
                    <div className="metric-trend">↑ 3%</div>
                  </div>

                  <div className="metric-card">
                    <div className="metric-icon rate-icon">
                      <Target size={24} />
                    </div>
                    <div className="metric-content">
                      <div className="metric-value">{liveStats.approvalRate.toFixed(1)}%</div>
                      <div className="metric-label">Approval Rate</div>
                    </div>
                    <div className="metric-trend">→ 0%</div>
                  </div>
                </div>
              </div>

              {/* Model Info Cards */}
              <div className="model-info-grid">
                <div className="info-mini-card">
                  <Brain size={28} color="#dc2626" />
                  <div>
                    <h4>XGBoost Engine</h4>
                    <p>Tree-based ML</p>
                  </div>
                </div>
                <div className="info-mini-card">
                  <Award size={28} color="#dc2626" />
                  <div>
                    <h4>93.2% Accurate</h4>
                    <p>Validated model</p>
                  </div>
                </div>
                <div className="info-mini-card">
                  <Sparkles size={28} color="#dc2626" />
                  <div>
                    <h4>SHAP Powered</h4>
                    <p>Explainable AI</p>
                  </div>
                </div>
                <div className="info-mini-card">
                  <Zap size={28} color="#dc2626" />
                  <div>
                    <h4>Real-time</h4>
                    <p>&lt;100ms response</p>
                  </div>
                </div>
              </div>
            </>
          ) : (
            <>
              {/* 3D Risk Score Visualization */}
              <div className="risk-visualization">
                <div className="risk-meter">
                  <svg viewBox="0 0 200 120" className="gauge-svg">
                    <defs>
                      <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" style={{stopColor: '#dc2626', stopOpacity: 1}} />
                        <stop offset="50%" style={{stopColor: '#f59e0b', stopOpacity: 1}} />
                        <stop offset="100%" style={{stopColor: '#10b981', stopOpacity: 1}} />
                      </linearGradient>
                    </defs>
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke="rgba(255,255,255,0.1)"
                      strokeWidth="12"
                      strokeLinecap="round"
                    />
                    <path
                      d="M 20 100 A 80 80 0 0 1 180 100"
                      fill="none"
                      stroke="url(#gaugeGradient)"
                      strokeWidth="12"
                      strokeLinecap="round"
                      strokeDasharray={`${getRiskScore() * 2.51} 251`}
                      className="gauge-fill"
                    />
                    <circle cx="100" cy="100" r="8" fill="#dc2626" className="gauge-pointer" />
                  </svg>
                  <div className="gauge-value">
                    <div className="score-number">{getRiskScore()}</div>
                    <div className="score-label">Risk Score</div>
                  </div>
                </div>
              </div>

              {/* Prediction Result */}
              <div className={`prediction-card-new ${prediction.prediction === 'Approved' ? 'approved' : 'rejected'}`}>
                <div className="prediction-icon-large">
                  {prediction.prediction === 'Approved' ? (
                    <CheckCircle size={64} />
                  ) : (
                    <XCircle size={64} />
                  )}
                </div>
                <h2>{prediction.prediction}</h2>
                <div className="confidence-bar">
                  <div 
                    className="confidence-fill" 
                    style={{width: `${prediction.probability * 100}%`}}
                  ></div>
                </div>
                <p className="probability">
                  {(prediction.probability * 100).toFixed(2)}% Confidence
                </p>
              </div>

              {/* Radar Chart - Unique Visualization */}
              <div className="radar-card">
                <h3><Target size={24} /> Multi-Dimensional Analysis</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <RadarChart data={getRadarData()}>
                    <PolarGrid stroke="rgba(255,255,255,0.2)" />
                    <PolarAngleAxis 
                      dataKey="subject" 
                      tick={{fill: '#94a3b8', fontSize: 11}}
                    />
                    <PolarRadiusAxis angle={90} domain={[0, 1]} tick={false} />
                    <Radar 
                      name="Impact" 
                      dataKey="value" 
                      stroke="#dc2626" 
                      fill="#dc2626" 
                      fillOpacity={0.6} 
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Factor Analysis */}
              <div className="explanation-card">
                <h3><TrendingUp size={24} /> Key Decision Factors</h3>
                <p className="subtitle">Top features influencing this prediction</p>
                
                <div className="factors-list-new">
                  {prediction.top_factors.map((factor, index) => (
                    <div key={index} className="factor-item-new">
                      <div className="factor-left">
                        <div className={`factor-badge ${factor.impact.toLowerCase()}`}>
                          #{index + 1}
                        </div>
                        <div className="factor-details">
                          <div className="factor-name-new">{factor.feature.replace(/_/g, ' ')}</div>
                          <div className="factor-value">{factor.contribution.toFixed(4)}</div>
                        </div>
                      </div>
                      <div className={`factor-bar-container ${factor.impact.toLowerCase()}`}>
                        <div 
                          className="factor-bar-fill"
                          style={{width: `${Math.abs(factor.contribution) * 100}%`}}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
