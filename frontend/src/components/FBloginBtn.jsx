import { useState, useEffect } from 'react';
// import FacebookLogin from 'react-facebook-login';
// import FacebookLogin from '@greatsumini/react-facebook-login';


const FBloginBtn = () => {
    const [accessToken, setAccessToken] = useState();
    const [userData, setUserData] = useState({})
    
    const responseFacebook = (response) => {
        console.log(response);
        setUserData({
            name: response.name,
            email: response.email,
            picture: response.picture.data.url,
        })
    }
    const componentClicked = data => {
        console.log("data", data);
    };
    // const responseFacebook = (response) => {
    //     console.log(response);
    // }

    return (
        <div>
            <br />
            User Short-Lived Access Token:
            <br />
            {accessToken}
            {/* {userData.name &&
                <div>
                    <img src={userData.picture} alt={userData.name} />
                    <p>Welcome, {userData.name}</p>
                    <p>Email: {userData.email}</p>
                </div>} */}
            <br />
            {/* <FacebookLogin
                appId="409932391961570"
                autoLoad={true}
                fields="name,email,picture"
                // onClick={componentClicked}
                callback={responseFacebook}
            /> */}
            
        </div>
    )
}

export default FBloginBtn;