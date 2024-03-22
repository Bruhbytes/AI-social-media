import './App.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Homepage from "./pages/Homepage";
// const backendUrl = process.env.REACT_APP_URL;
const backendUrl = "https://ai-social-media-server.vercel.app"

function App() {  
  const [user, setUser] = useState(null);

  const getUser = async () => {
    try {
      // const url = `${backendUrl}/auth/login/success`;
      const url = `${process.env.REACT_APP_URL}/auth/login/success`;
      const { data } = await axios.get(url, { withCredentials: true });
      setUser(data.user._json);
      console.log(data);
    } catch (err) {
      console.log(err);
    }
  };

  useEffect(() => {
    getUser();
  }, [])

  return (
    <div className="App">
      <Routes>
        <Route exact path="/" element={<Homepage />} />
        <Route
          exact
          path="/login"
          element={user ? <Navigate to="/" /> : <Login />}
        />
        <Route
          path="/signup"
          element={user ? <Navigate to="/" /> : <Signup />}
        />
      </Routes>
    </div>
  );
}

export default App;
