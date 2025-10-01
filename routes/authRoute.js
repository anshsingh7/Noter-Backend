const express = require("express");
const {
  registerUser,
  loginUser,
  updateFacePattern,
  getCurrentUser,
  adminGetAllUserController,
  moderatorAllUsersController,
  loginUserWithFace,
} = require("../controllers/authController");
const { protect, authorizeRoles } = require("../middleware/authMiddleware");
const upload = require("../utils/utilityFunction");

const router = express.Router();

// Public
router.post("/register", registerUser);
router.post("/login", loginUser);
router.post("/login-through-face", loginUserWithFace);
router.put("/update/:id", protect,updateFacePattern);

// Current user
router.get("/currentUser", protect, getCurrentUser);

// Admin only
router.get("/admin/getAllUsers", protect, authorizeRoles("admin"), adminGetAllUserController);

// Moderator only
router.get("/moderator/getAllUsers", protect, authorizeRoles("moderator"), moderatorAllUsersController);

module.exports = router;
