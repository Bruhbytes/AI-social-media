import React from "react";
import img1 from "../../data/img11.jpg";
import img2 from "../../data/img12.jpg";
import img3 from "../../data/img13.jpg";
import img4 from "../../data/img14.jpg";
import img5 from "../../data/img15.jpg";
import img6 from "../../data/img17.jpg";
import img7 from "../../data/img16.jpg";
import './Analysis.css';

const Analysis = () => {
  // Sample card data with imported images
  const cardData = [
    {
      id: 1,
      image: img1, // Import the image
      title: "TOTAL ORDERS BY PRODUCT TYPE",
      description:
        "To improve for users, focus on expanding the sandals and sneakers range as they are the highest selling categories. Additionally, explore marketing strategies to boost sales of lower demand products like heels and boots to better cater to diverse customer preferences.",
    },
    {
      id: 2,
      image: img2, // Import the image
      title: "TOTAL ORDERS BY BRANDS",
      description:
        "To meet customer needs effectively, it's essential to improve the product variety and promotional tactics for popular brands such as Bata, Paragon, and Adidas. Furthermore, studying customer preferences for specialized brands like Liberty, Woodland, and Reebok could reveal potential avenues for extending their influence and attractiveness.",
    },
    {
      id: 3,
      image: img3, // Import the image
      title: "EVALUATION OF TOTAL ORDERS",
      description:
        "The graph displays order evaluations across 2023, peaking notably in January and December, indicating periods of heightened purchasing activity likely influenced by seasonal trends or holiday festivities.",
    },
    {
      id: 4,
      image: img4, // Import the image
      title: "ORDER PURCHSE DAY OF WEEK",
      description:
        "Analyzing the graph reveals higher order volumes on Saturdays, prompting strategic adjustments to optimize operations and maximize revenue potential across different days.",
    },
    {
      id: 5,
      image: img5, // Import the image
      title: "PROFIT BY PRODUCT TYPE",
      description:
        "To boost profits, focus resources on high-margin categories like flats and sandals, while also developing strategies to improve margins for lower-performing ones such as heels and boots, leading to optimized revenue and profitability through ongoing profit analysis and strategic decision-making.",
    },
    {
      id: 6,
      image: img6, // Import the image
      title: "PROFIT BY BRAND",
      description:
        "To maximize profitability, prioritize promoting top-profit-margin brands like Nike, Paragon, and Bata, while analyzing and improving profitability for lower-margin brands like Metro, Puma, and Liberty through pricing, cost optimizations, or targeted marketing, ensuring ongoing brand mix adjustments for enhanced revenue and profitability.",
    },
    {
      id: 7,
      image: img7, // Import the image
      title: "PROFIT BY BRAND",
      description:
        "To maximize profitability, prioritize promoting top-profit-margin brands like Nike, Paragon, and Bata, while analyzing and improving profitability for lower-margin brands like Metro, Puma, and Liberty through pricing, cost optimizations, or targeted marketing, ensuring ongoing brand mix adjustments for enhanced revenue and profitability.",
    },
  ];

  return (
    <div className="card-container">
      {cardData.map((card) => (
        <div key={card.id} className="card">
          <div className="card-content">
            <img
              src={card.image}
              alt={card.title}
              className="card-image"
            />
          </div>
          <h3 className="card-title">{card.title}</h3>
          <p className="card-description">{card.description}</p>
        </div>
      ))}
    </div>
  );
};

export default Analysis;
