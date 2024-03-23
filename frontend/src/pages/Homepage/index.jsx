import React from 'react';
import { Link } from 'react-router-dom';
import './home.css'; // Import external CSS file

const Homepage = () => {
  return (
    <div className='bigCont'>
    <div className="mycontainer">
      <div className="navbar">
        <div className="company-name">Social Spark</div>
        <div className="links">
          <Link to="/login" className=" login">Login</Link>
          <Link to="/signup" className=" signup">Signup</Link>
        </div>
      </div>
    </div>
      <div className="tab">
        <p>"Social Spark" is your go-to platform for unleashing the full potential of your online presence</p>
        <hr />
        <p>Access the below tool to generate some fun captions for your post</p>
        <Link to="/generate" className=" tab">Caption Generator</Link>
      </div>
    </div>
  );
};

export default Homepage;
