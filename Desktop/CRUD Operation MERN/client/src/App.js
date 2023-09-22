import "./App.css";
import AppNavbar from "./components/AppNavBar";
import Addusers from "./components/AddUser";
import Allusers from "./components/AllUsers";
import NewHome from "./components/NewHome"; // Adjust the path if necessary

import { BrowserRouter, Routes, Route } from "react-router-dom";

function App() {
  return (
    <BrowserRouter>
      <AppNavbar />
      <Routes>
        <Route path="/" element={<NewHome />} />
        <Route path="/add" element={<Addusers />} />
        <Route path="/all" element={<Allusers />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
