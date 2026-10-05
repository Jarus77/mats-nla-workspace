// Assembles the deck content in order. Edit the section files; this just concatenates them.
const meta = require("./00_meta");

const slides = [
  ...require("./01_intro"),
  ...require("./02_design"),
  ...require("./03_gates"),
  ...require("./05_partA"),
  ...require("./06_partB"),
  ...require("./06b_partC"),
  ...require("./04_wsbench"),
  ...require("./07_close"),
  ...require("./08_appendix"),
];

module.exports = { ...meta, slides };
