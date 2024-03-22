const router = require("express").Router();
const passport = require("passport");
const frontendUrl = "https://ai-social-media-client.vercel.app"

router.get("/login/success", (req, res) => {
  if (req.user) {
    res.status(200).json({
      error: false,
      message: "Successfully Loged In",
      user: req.user,
    });
  } else {
    res.status(403).json({ error: true, message: "Not Authorized" });
  }
});

router.get("/login/failed", (req, res) => {
  res.status(401).json({
    error: true,
    message: "Log in failure",
  });
});

router.get("/google", passport.authenticate("google", ["profile", "email"]));

router.get(
  "/google/callback",
  passport.authenticate("google", {
    // successRedirect: process.env.FRONTEND_URL,
    successRedirect: frontendUrl,
    failureRedirect: "/login/failed",
  })
);

router.get("/logout", (req, res) => {
  req.logout();
  // res.redirect(process.env.FRONTEND_URL);
  res.redirect(frontendUrl);
});

module.exports = router;
