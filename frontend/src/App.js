import './App.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
// const backendUrl = process.env.REACT_APP_URL;
const backendUrl = "https://ai-social-media-server.vercel.app"

function App() {
  const [data, setData] = useState(null);

  useEffect(() => {
    const fetchData = () => {
      axios.get(`${backendUrl}/`)
        .then(async (response) => {
          if (response.status !== 200) {
            console.log("Could not get data");
          }
          else {
            const jsonData = await response.data;
            setData(jsonData);
            console.log(jsonData)
          }
        })
    }

    fetchData();
  }, [])

  return (
    <div className="App">
      <h1>Hello</h1>
      {data && <p>{data.msg}</p>}
    </div>
  );
}

export default App;
