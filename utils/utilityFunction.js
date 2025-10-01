// middleware/uploadMiddleware.js
const multer = require("multer");
const path = require("path");
const bcrypt = require("bcryptjs");
const jwt = require("jsonwebtoken");
require("dotenv").config();

const JWT_SECRET = process.env.JWT_SECRET;
const JWT_EXPIRE = process.env.JWT_EXPIRE || 2592000 // 30 days in seconds; 

// Generate JWT
const generateToken = (userId, role) => {
  return jwt.sign({ id: userId, role }, JWT_SECRET, { expiresIn: JWT_EXPIRE });
};

// Hash Password
const hashPassword = async (password) => {
  const salt = await bcrypt.genSalt(10); // 🔹 explicitly added salt
  let hashedPassword = await bcrypt.hash(password, salt);

  return hashedPassword;
};

// Compare Password
const comparePassword = async (enteredPassword, hashedPassword) => {
  let isMatch = await bcrypt.compare(enteredPassword, hashedPassword);

  return isMatch
};


// Storage config
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/"); // save to uploads folder
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    cb(null, Date.now() + ext); // unique filename
  },
});

const upload = multer({ storage });

module.exports = upload;


module.exports = {
  generateToken,
  hashPassword,
  comparePassword,
};
