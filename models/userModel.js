const mongoose = require("mongoose");
const { Schema } = mongoose;

const userSchema = new Schema(
  {
    id: {
      type: Schema.Types.ObjectId,
      default: () => new mongoose.Types.ObjectId(),
      unique: true,
    },
    firstName: {
      type: String,
      required: [true, "First Name is required"],
      minlength: [2, "First Name must be at least 2 characters long"],
      maxlength: [50, "First Name cannot exceed 50 characters"],
    },
    lastName: {
      type: String,
      required: [true, "Last Name is required"],
      minlength: [2, "Last Name must be at least 2 characters long"],
      maxlength: [50, "Last Name cannot exceed 50 characters"],
    },
    email: {
      type: String,
      required: [true, "Email is required"],
      unique: true,
      lowercase: true,
      trim: true,
      match: [
        /^\w+([\.-]?\w+)*@\w+([\.-]?\w+)*(\.\w{2,})+$/,
        "Please enter a valid email address",
      ],
    },

    password: {
      type: String,
      required: [true, "Password is required"],
      minlength: [10, "Password must be at least 6 characters"],
      match: [
        /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)(?=.*[\W_]).{6,}$/,
        "Password must contain at least one uppercase letter, one lowercase letter, one number, and one special character",
      ],
      select: false, // don’t return password by default
    },

    role: {
      type: String,
      enum: ["user", "admin", "moderator"],
      default: "user",
    },
    photo: { type: String }, // path or URL to uploaded image
    faceEncodedData: { type: [Number] },

    notes: [{ type: Schema.Types.ObjectId, ref: "Note" }],
    tasks: [{ type: Schema.Types.ObjectId, ref: "Task" }],
  },
  { timestamps: true }
);

module.exports = mongoose.model("User", userSchema);
