import React from 'react';
import { Link } from 'react-router-dom';
import './home.css'; // Import external CSS file

const Homepage = () => {
  return (
    <div className="container">
      <div className="navbar">
        <div className="company-name">GitRitz</div>
        <div className="nav-links">
          <Link to="/login" className="nav-link login">Login</Link>
          <Link to="/signup" className="nav-link signup">Signup</Link>
        </div>
      </div>
      <div className="tab">
        <Link to="/caption-generator" className="nav-link tab">Caption Generator</Link>
      </div>
    </div>
  );
};

export default Homepage;
