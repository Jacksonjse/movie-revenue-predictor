import React, { useState } from "react";
import axios from "axios";

function App() {
  const [formData, setFormData] = useState({
    budget: "",
    runtime: "",
    popularity: "",
    vote_average: "",
    vote_count: "",
  });
  const [prediction, setPrediction] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const res = await axios.post("http://localhost:8000/predict", formData);
      setPrediction(res.data.predicted_revenue);
    } catch (err) {
      console.error(err);
      alert("Error connecting to backend");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "40px" }}>
      <h1>🎬 Movie Revenue Predictor</h1>
      <form
        onSubmit={handleSubmit}
        style={{
          display: "flex",
          flexDirection: "column",
          width: "300px",
          margin: "0 auto",
          gap: "10px",
        }}
      >
        {Object.keys(formData).map((key) => (
          <input
            key={key}
            type="number"
            name={key}
            value={formData[key]}
            onChange={handleChange}
            placeholder={key}
            required
          />
        ))}
        <button type="submit">Predict</button>
      </form>

      {prediction && (
        <h2 style={{ marginTop: "20px" }}>
          Predicted Revenue: ${prediction.toFixed(2)}
        </h2>
      )}
    </div>
  );
}

export default App;
