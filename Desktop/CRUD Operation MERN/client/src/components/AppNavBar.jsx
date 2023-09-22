import { AppBar, Toolbar, styled } from "@mui/material";
import { NavLink } from "react-router-dom";

const Header = styled(AppBar)`
  background: #111111;
`;

const Tabs = styled(NavLink)`
  font-size: 20px;
  margin-right: 50px;
  color: inherit;
`;

const AppNavbar = () => {
  return (
    <Header position="static">
      <Toolbar>
        <Tabs to="/home">Home</Tabs>
        <Tabs to="/all">All Users</Tabs>
        <Tabs to="/add">Add Users</Tabs>
      </Toolbar>
    </Header>
  );
};

export default AppNavbar;
