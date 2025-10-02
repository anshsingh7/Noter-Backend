const User = require("../models/userModel");
const axios = require("axios");
const { generateToken, hashPassword, comparePassword } = require("../utils/utilityFunction");

// ✅ CURRENT USER
const getCurrentUser = async (req, res) => {
  try {
    if (!req.user) {
      return res.status(401).json({ message: "Not authorized" });
    }

    res.json({
      success: true,
      message: "Welcome Admin Dashboard",
      user: {
        id: req.user.id,
        firstName: req.user.firstName,
        lastName: req.user.lastName,
        email: req.user.email,
        role: req.user.role,
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};

// ✅ REGISTER USER
const registerUser = async (req, res) => {
  try {
    const { firstName, lastName, email, password, role } = req.body;

    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: "User already exists" });
    }

    const hashedPassword = await hashPassword(password);

    const user = await User.create({
      firstName,
      lastName,
      email,
      password: hashedPassword,
      role: role || "user",
    });

    res.status(201).json({
      success: true,
      message: "User registered successfully",
      token: generateToken(user._id, user.role),
      user: {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};

// ✅ LOGIN USER
const loginUser = async (req, res) => {
  try {
    const { email, password } = req.body;

    
    
    const user = await User.findOne({ email }).select("+password");
    if (!user) {
      return res.status(401).json({ message: "Invalid credentials" });
    }
    
    const isMatch = await comparePassword(password, user.password);
    if (!isMatch) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    res.json({
      success: true,
      token: generateToken(user._id, user.role),
      user: {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        encoding: user.faceEncodedData || [],
      },
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};


const updateFacePattern = async (req, res) => {
  try {
    const { id } = req.params;
    
    const pythonUrl = process.env.PYTHON_API_URL || "http://localhost:5000";
    const updatedUser = await User.findById(id);

    if (!updatedUser) {
      return res.status(404).json({ success: false, message: "User not found" });
    }


    // Call Python backend
    const pythonRes = await axios.post(
      `${pythonUrl}/capture-face`,
      { userId: id },
      { timeout: 2 * 60 * 1000 } // 2 min timeout
    );

    if (!pythonRes.data || !pythonRes.data.success) {
      return res.status(400).json({
        success: false,
        message: (pythonRes.data && pythonRes.data.message) || "Python face capture failed",
      });
    }

    // ✅ Extract encoding
    const encoding = pythonRes.data.encoding;
    const finalEncoding = (encoding && encoding.length > 0) ? encoding : [];

    // ✅ Update user in MongoDB
    updatedUser.faceEncodedData = finalEncoding;
    await updatedUser.save();

    return res.json({
      success: true,
      message: "Face pattern updated (Python backend).",
      user: updatedUser,
    });
  } catch (error) {
    console.error("updateFacePattern error:", error.message || error);
    if (error.response && error.response.data) {
      return res.status(500).json({ success: false, message: "Python error", error: error.response.data });
    }
    return res.status(500).json({ success: false, message: "Server error", error: error.message });
  }
};

const loginUserWithFace = async (req, res) => {
  try {
    const { email } = req.body;

    if (!email) {
      return res.status(400).json({ message: "Email is required for face login" });
    }

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(401).json({ message: "Invalid credentials" });
    }

    if (!user.faceEncodedData || user.faceEncodedData.length === 0) {
      return res.status(400).json({ message: "No face data registered for this user" });
    }

    const pythonUrl = process.env.PYTHON_API_URL || "http://localhost:5000";

    // 🔑 Call Python backend
    const pythonRes = await axios.post(
      `${pythonUrl}/verify-face`,
      { encoding: user.faceEncodedData },
      { timeout: 2 * 60 * 1000 }
    );

    if (!pythonRes.data || !pythonRes.data.success) {
      return res.status(400).json({
        success: false,
        message: (pythonRes.data && pythonRes.data.message) || "Face verification failed",
      });
    }

    // ✅ Check match percent
      if (!pythonRes.data.match ) {
      return res.status(401).json({
        success: false,
        message: "Face did not match",
        similarity: pythonRes.data.similarity,
        matchPercent: pythonRes.data.matchPercent,
      });
    }

    // ✅ If match → return same response as password login
    return res.json({
      success: true,
      token: generateToken(user._id, user.role),
      user: {
        id: user.id,
        firstName: user.firstName,
        lastName: user.lastName,
        email: user.email,
        role: user.role,
        encoding: user.faceEncodedData || [],
      },
      similarity: pythonRes.data.similarity,
      matchPercent: pythonRes.data.matchPercent,
    });
  } catch (error) {
    console.error("Face login error:", error.message || error);
    if (error.response && error.response.data) {
      return res.status(500).json({
        success: false,
        message: "Python error",
        error: error.response.data,
      });
    }
    return res.status(500).json({ success: false, message: "Server error", error: error.message });
  }
};




// ✅ ADMIN CONTROLLER
// controllers/adminController.js

const adminGetAllUserController = async (req, res) => {
  try {

    // Fetch all users
    const users = await User.find({}); // get all users

    res.json({
      success: true,
      message: "Welcome Admin Dashboard",
      admin: {
        id: req.user.id,
        name: `${req.user.firstName} ${req.user.lastName}`,
        email: req.user.email,
        role: req.user.role,
      },
      users, // return all users here
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};


// ✅ MODERATOR CONTROLLER
const moderatorAllUsersController = async (req, res) => {
  try {

    
    const users = await User.find({role: ["user", "moderator"] }); // get all users except admins

    res.json({
      success: true,
      message: "Welcome moderator Dashboard",
      admin: {
        id: req.user.id,
        name: `${req.user.firstName} ${req.user.lastName}`,
        email: req.user.email,
        role: req.user.role,
      },
      users, // return all users here
    });
  } catch (error) {
    res.status(500).json({ message: "Server Error", error: error.message });
  }
};

module.exports = {
  getCurrentUser,
  registerUser,
  loginUser,
  loginUserWithFace,
  updateFacePattern,
  adminGetAllUserController,
  moderatorAllUsersController,
};
