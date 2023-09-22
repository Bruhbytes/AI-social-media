import {
  FormControl,
  FormGroup,
  InputLabel,
  Input,
  styled,
  Typography,
  Button,
} from "@mui/material";

import { useState } from "react";
const Container = styled(FormGroup)`
  width: 50%;
  margin: 5% auto 0 auto;
  & > div {
    margin-top: 20px;
  }
`;

const DefaultValue = {
  name: "",
  username: "",
  email: "",
  phone: "",
};

const Addusers = () => {
  const [user, setUser] = useState();

  return (
    <Container>
      <Typography variant="h4"> ADD User</Typography>
      <FormControl>
        <InputLabel>Name</InputLabel>
        <Input />
      </FormControl>
      <FormControl>
        <InputLabel>Username</InputLabel>
        <Input />
      </FormControl>
      <FormControl>
        <InputLabel>Phone</InputLabel>
        <Input />
      </FormControl>
      <FormControl>
        <InputLabel>Email</InputLabel>
        <Input />
      </FormControl>
      <FormControl>
        <Button variant="contained">ADD USER</Button>
      </FormControl>
    </Container>
  );
};

export default Addusers;
