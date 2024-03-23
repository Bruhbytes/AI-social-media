// import React, { useState } from "react";
// import axios from "axios";

// function App() {
//   const [startDate, setStartDate] = useState("");
//   const [endDate, setEndDate] = useState("");
//   const [bestTimes, setBestTimes] = useState("");

//   const handleSubmit = async (e) => {
//     e.preventDefault();
//     try {
//       const response = await axios.post("http://localhost:5000/predict", {
//         start_date: startDate,
//         end_date: endDate,
//       });
//       setBestTimes(response.data.best_times);
//     } catch (error) {
//       console.error("Error:", error);
//     }
//   };

//   return (
//     <div>
//       <h1>Instagram Reach Forecast</h1>
//       <form onSubmit={handleSubmit}>
//         <label>Start Date:</label>
//         <input
//           type="date"
//           value={startDate}
//           onChange={(e) => setStartDate(e.target.value)}
//         />
//         <label>End Date:</label>
//         <input
//           type="date"
//           value={endDate}
//           onChange={(e) => setEndDate(e.target.value)}
//         />
//         <button type="submit">Predict Best Times</button>
//       </form>
//       {bestTimes && <p>Best times to upload: {bestTimes}</p>}
//     </div>
//   );
// }

// export default App;

import React, { useState } from "react";
import axios from "axios";

function Schedule() {
  const [startDate, setStartDate] = useState("");
  const [endDate, setEndDate] = useState("");
  const [bestCampaignDate, setBestCampaignDate] = useState("");
  const [expectedCTR, setExpectedCTR] = useState("");

  const handleStartDateChange = (e) => {
    setStartDate(e.target.value);
  };

  const handleEndDateChange = (e) => {
    setEndDate(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.get(
        "http://localhost:6000/best_campaign_date",
        {
          params: {
            start_date: startDate,
            end_date: endDate,
          },
        }
      );
      setBestCampaignDate(response.data.best_campaign_date);
      setExpectedCTR(response.data.expected_ctr);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="App">
      <h1 className="head">Marketing Campaign Analyzer</h1>
      <form onSubmit={handleSubmit}>
        <label>
          Start Date:
          <input
            type="date"
            value={startDate}
            onChange={handleStartDateChange}
          />
        </label>
        <label>
          End Date:
          <input type="date" value={endDate} onChange={handleEndDateChange} />
        </label>
        <button type="submit">Submit</button>
      </form>

      <div>
        <h2>Best Campaign Date: {bestCampaignDate}</h2>
        <h2>Expected CTR: {expectedCTR}</h2>
      </div>
    </div>
  );
}

export default Schedule;
