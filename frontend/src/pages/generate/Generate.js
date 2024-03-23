import React, { useState } from "react";
import axios from "axios"; // Import Axios library
import "./Generate.css";

const App = () => {
  const [body, setbody] = useState("");
  const [responseData, setResponseData] = useState(null);
  const [error, setError] = useState(null);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setbody(value);
  };

  const handlePostRequest = async (e) => {
    e.preventDefault();
    console.log(body);
    try {
      axios
        .post(
          "http://localhost:8000/chat",
          { query: body },
          { headers: { "Content-Type": "application/json" } }
        )
        .then(function (response) {
          if (response.status === 200) {
            console.log(response.data);
            const data = response.data;
            setResponseData(data);
          } else console.log("Error");
        })
        .catch(function (error) {
          console.log(error.response.data);
        });
    } catch (error) {
      setError(error.message);
    }
  };

  return (
    <div className="app-container">
      <h1>
        <b>Talk to Chat bot for any help!</b>
      </h1>
      <div className="form-container">
        <label>USER:</label>
        <textarea
          name="body"
          value={body}
          onChange={handleInputChange}
        ></textarea>
        <button onClick={handlePostRequest}>Send Post Request</button>
      </div>

      {responseData && (
        <div className="response-container">
          <h2>
            <b>RESPONSE DATA:</b>
          </h2>
          <pre>{JSON.stringify(responseData, null, 2)}</pre>
        </div>
      )}
    </div>
  );
};

export default App;
