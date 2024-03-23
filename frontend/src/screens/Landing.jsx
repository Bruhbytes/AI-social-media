import React from "react";
// Sections

import TopNavbar from "../Nav/TopNavbar"
import Header from "../Sections/Header";
import Services from "../Sections/Services";

import Contact from "../Sections/Contact";
import Footer from "../Sections/Footer"

export default function Landing() {
  return (
    <>
      <TopNavbar />
      <Header />
      <Services />      
      <Contact />
      <Footer />
    </>
  );
}


