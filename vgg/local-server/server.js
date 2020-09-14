const express = require("express");
let app = express();

app.use(function (req, res, next) {
 console.log(`${new Date()} - ${req.method} request for ${req.url}`);
  next();
});
app.use(express.static("../static"));


app.listen(81, () => {
  console.log(
    `Server static on 81`
  );
});
