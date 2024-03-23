import { Link } from "react-router-dom";
import styles from "./login.css";


const backendUrl = "https://ai-social-media-server.vercel.app"

function Login() {
	const googleAuth = () => {
		// window.open(
		// 	`${backendUrl}/auth/google/callback`,
		// 	"_self"
		// );
		window.open(
			`${process.env.REACT_APP_URL}/auth/google/callback`,
			"_self"
		);
	};
	return (
		<div className="container">
			<h1 className="heading">Log in Form</h1>
			<div className="form_container">
				<div className="left">
					<img className="img" src="./images/login.jpg" alt="login" />
				</div>
				<div className="right">
					<h2 className="from_heading">Members Log in</h2>
					<input type="text" className="input" placeholder="Email" />
					<input type="text" className="input" placeholder="Password" />
					<button className="btn">Log In</button>
					<p className="text">or</p>
					<button className="google_btn" onClick={googleAuth}>
						<img src="./images/google.png" alt="google icon" />
						<span>Sign in with Google</span>
					</button>					
					<p className="text">
						New Here ? <Link to="/signup">Sign Up</Link>
					</p>
				</div>
			</div>
		</div>
	);
}

export default Login;