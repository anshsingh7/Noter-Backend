const express = require("express");
const dotenv = require("dotenv");
const cors = require("cors");
const authRoutes = require("./routes/authRoute");
const dbConnect = require("./db/db");

dotenv.config();
dbConnect();


let app = express();

// ✅ MiddlewaresF
app.use(cors());
app.use(express.json()); // <-- parse application/json
app.use(express.urlencoded({ extended: true }));  // parse form data


app.use("/v1/auth",authRoutes);

let port = process.env.PORT||3001
app.listen(port,()=>console.log(`connect to port : ${port}`))