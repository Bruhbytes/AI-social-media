import './App.css';
import { useEffect, useState } from 'react';
import axios from 'axios';
import { Routes, Route, Navigate } from "react-router-dom";
import Home from "./pages/Home";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Homepage from "./pages/Homepage";
import Home2 from './pages/Home2';
import Analysis from './pages/Analysis/Analysis';
import Plagia from './pages/Plagia/Plagia';
import Generate from './pages/generate/Generate';
import Schedule from './pages/Schedule/Schedule';

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
          element={user ? <Navigate to="/services" /> : <Login />}
        />
        <Route
          path="/signup"
          element={user ? <Navigate to="/services" /> : <Signup />}
        />
        <Route
          path="/services"
          element={user ? <Home2 /> : <Login />}
        />
        <Route
          path="/services/analysis"
          element={user ? <Analysis /> : <Login />}
        />
        <Route
          path="/services/plagia"
          element={user ? <Plagia /> : <Login />}
        />
        <Route
          path="/generate"
          element= {<Generate />}
        />
        <Route
          path="/services/schedule"
          element={user ? <Schedule /> : <Login />}
        />
      </Routes>
    </div>
  );
}

export default App;
