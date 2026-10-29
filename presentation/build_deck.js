// Section C · Team 17 — 3D FoJ presentation deck
const pptxgen = require("pptxgenjs");

const FIGS = "/Users/samudraneelsarkar/Downloads/GitHub/IPCV Project/report/figures";

const DARK = "21295C"; // midnight — title/conclusion background
const PRIMARY = "065A82"; // deep blue — titles
const SECONDARY = "1C7293"; // teal
const ACCENT = "F4A261"; // amber — stat callouts
const CARD = "EEF3F7"; // light card fill
const TEXT = "1B2A33"; // body
const MUTED = "5E7183"; // captions
const OKGREEN = "2E9E5B";
const BADRED = "C4452F";
const LIGHT_ON_DARK = "C9D4E8";

const TITLE_FONT = "Cambria";
const BODY_FONT = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
const W = 13.333;
const MX = 0.6; // side margin

function footer(slide, n) {
  slide.addText("3D FoJ · Section C Team 17", {
    x: MX, y: 7.12, w: 4, h: 0.3, margin: 0,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED,
  });
  slide.addText(String(n), {
    x: W - 1.0, y: 7.12, w: 0.4, h: 0.3, margin: 0,
    fontFace: BODY_FONT, fontSize: 9, color: MUTED, align: "right",
  });
}

function header(slide, title, kicker) {
  if (kicker) {
    slide.addText(kicker.toUpperCase(), {
      x: MX, y: 0.32, w: 12.1, h: 0.3, margin: 0,
      fontFace: BODY_FONT, fontSize: 11, bold: true, color: SECONDARY,
      charSpacing: 2,
    });
  }
  slide.addText(title, {
    x: MX, y: 0.62, w: 12.1, h: 0.75, margin: 0,
    fontFace: TITLE_FONT, fontSize: 30, bold: true, color: PRIMARY,
  });
}

function card(slide, x, y, w, h, title, body, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, {
    x, y, w, h, rectRadius: 0.09,
    fill: { color: opts.fill || CARD },
    line: opts.line || { color: "D5DEE6", width: 1 },
  });
  slide.addText(
    [
      { text: title, options: { bold: true, fontSize: opts.titleSize || 15, color: opts.titleColor || PRIMARY, breakLine: true } },
      { text: body, options: { fontSize: opts.bodySize || 12.5, color: TEXT } },
    ],
    {
      x: x + 0.22, y: y + 0.16, w: w - 0.44, h: h - 0.32, margin: 0,
      fontFace: BODY_FONT, valign: "top", paraSpaceAfter: 6,
    }
  );
}

function bullets(slide, items, x, y, w, h, opts = {}) {
  const runs = items.map((t, i) => ({
    text: t,
    options: {
      bullet: { indent: 14 },
      breakLine: true,
      paraSpaceAfter: opts.gap == null ? 10 : opts.gap,
      fontSize: opts.fontSize || 14,
      color: opts.color || TEXT,
    },
  }));
  slide.addText(runs, {
    x, y, w, h, margin: 0, fontFace: BODY_FONT, valign: "top",
  });
}

function stat(slide, x, y, w, value, label, opts = {}) {
  slide.addText(value, {
    x, y, w, h: 0.85, margin: 0, align: "center",
    fontFace: TITLE_FONT, fontSize: opts.size || 44, bold: true,
    color: opts.color || ACCENT,
  });
  slide.addText(label, {
    x, y: y + 0.9, w, h: 1.0, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 12, color: opts.labelColor || TEXT,
    valign: "top",
  });
}

/* ---------------- Slide 1 — title (dark) ---------------- */
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addText("What Determines the Effectiveness of\n3D Field of Junctions?", {
    x: 0.8, y: 1.7, w: W - 1.6, h: 1.9, margin: 0, align: "center",
    fontFace: TITLE_FONT, fontSize: 38, bold: true, color: "FFFFFF",
  });
  s.addText("A Study of Volumetric Denoising Applications", {
    x: 0.8, y: 3.75, w: W - 1.6, h: 0.5, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 20, italic: true, color: ACCENT,
  });
  s.addText("SECTION C · TEAM 17", {
    x: 0.8, y: 4.85, w: W - 1.6, h: 0.4, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: "FFFFFF",
    charSpacing: 3,
  });
  s.addText(
    "Samudraneel Sarkar · Sakshi Patil · Nikhil Kumar · Manpreet Singh",
    {
      x: 0.8, y: 5.35, w: W - 1.6, h: 0.35, margin: 0, align: "center",
      fontFace: BODY_FONT, fontSize: 13, color: LIGHT_ON_DARK,
    }
  );
  s.addText("MNNIT Allahabad   ·   co-authors contribute equally", {
    x: 0.8, y: 5.75, w: W - 1.6, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 11, color: "8FA8C8",
  });
  s.addText("github.com/samudraneel05/SectionC_Team17", {
    x: 0.8, y: 6.9, w: W - 1.6, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 11, color: "8FA8C8",
  });
  s.addNotes(
    "Introduce the team and the question in one sentence: the paper showed 3D FoJ denoising CT, cryo-ET, and point clouds — we ask WHEN it works, not whether it works."
  );
}

/* ------------- Slide 2 — the method ------------- */
{
  const s = pres.addSlide();
  header(s, "3D Field of Junctions, in one slide", "Background");
  s.addImage({
    path: `${FIGS}/ext/fig_junction_full.png`,
    x: (W - 11) / 2, y: 1.55, w: 11, h: 11 * (340 / 2173),
  });
  s.addText("Junction schematic from the original paper (Moeini et al., ECCV 2026)", {
    x: MX, y: 3.35, w: 12.1, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });
  const cw = 3.87, gap = 0.25, y0 = 3.95, ch = 2.15;
  card(s, MX, y0, cw, ch, "Training-free",
    "No network, no dataset, no training. The model is fit directly on the single noisy input volume.");
  card(s, MX + cw + gap, y0, cw, ch, "Wedges per patch",
    "Every R\u00B3 cubic patch is split by planes through one shared vertex into wedge regions; each region takes the mean of the voxels it covers.");
  card(s, MX + 2 * (cw + gap), y0, cw, ch, "Overlap averaging",
    "Patches tile the volume on a grid. Each voxel is rendered by several patches and the outputs are averaged into the final volume.");
  s.addText("Moeini, Kim, Romberg & Fridovich-Keil — released implementation used as-is", {
    x: MX, y: 6.72, w: 12.1, h: 0.3, margin: 0,
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });
  footer(s, 2);
  s.addNotes(
    "Mechanism in three steps: independent patches, each patch becomes a junction of constant wedges, overlaps averaged. Emphasize training-free — that's the appeal vs. learned denoisers."
  );
}

/* ------------- Slide 3 — paper's domains vs ours ------------- */
{
  const s = pres.addSlide();
  header(s, "What the paper demonstrated — and what we test", "Motivation");
  s.addImage({
    path: `${FIGS}/ext/cvpr_teaser.png`,
    x: MX, y: 1.55, w: 5.9, h: 5.9 * (812 / 2034),
  });
  s.addText("Original teaser: low-dose CT, cryo-ET, point clouds", {
    x: MX, y: 4.0, w: 5.9, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });
  bullets(s, [
    "Demonstrated on sparse-view CT, cryo-electron tomography, and point-cloud denoising.",
    "Open question: does the wedge prior transfer to volumes with different structure?",
    "We apply the released implementation, unchanged, to three new domains.",
  ], MX, 4.55, 5.9, 2.2, { fontSize: 13.5, gap: 8 });

  const x = 7.0, w = 5.7, ch = 1.5, gap = 0.24;
  card(s, x, 1.55, w, ch, "Bentheim sandstone micro-CT",
    "Grains and pores separated by sharp interfaces, 10\u201335 voxels across — exactly the shape the wedge prior assumes.",
    { fill: "E9F4EF", line: { color: OKGREEN, width: 1 } });
  card(s, x, 1.55 + ch + gap, w, ch, "BrainWeb T1 MRI phantom",
    "Anatomical tissue has smooth gradients, not plateaus — tests the piecewise-flat assumption.",
    { fill: "FDF0EA", line: { color: "D9A08A", width: 1 } });
  card(s, x, 1.55 + 2 * (ch + gap), w, ch, "Binary porous media",
    "Sphere pack + Berea scan with ~2-voxel features, smaller than any patch — tests the scale assumption.",
    { fill: "FDF0EA", line: { color: "D9A08A", width: 1 } });
  footer(s, 3);
  s.addNotes(
    "Deliberate domain design: one domain built like the prior assumes (Bentheim), two that violate it in different ways (smooth MRI, sub-patch binary). This is what makes the results a study of conditions, not just a benchmark."
  );
}

/* ------------- Slide 4 — effective estimator / requirements ------------- */
{
  const s = pres.addSlide();
  header(s, "The effective estimator — and three requirements", "Mechanism");
  card(s, MX, 1.5, 5.95, 1.65, "The nominal model",
    "A coupled field of junctions: boundary- and color-consistency terms tie overlapping patches together, weighted by \u03BB ramps over refinement iterations.");
  card(s, MX + 6.2, 1.5, 5.95, 1.65, "What runs at the released defaults",
    "With N_ref = 1 the consistency weights never engage, and the single refinement pass renders from pre-update parameters. Effectively: an independent wedge fit per patch + overlap averaging.",
    { fill: "FBF3E4", line: { color: ACCENT, width: 1.25 } });
  s.addText("Three requirements for FoJ to help — all follow from \u201Creplace each wedge by its mean\u201D:", {
    x: MX, y: 3.55, w: 12.1, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 13.5, italic: true, color: TEXT,
  });
  const cw = 3.87, gap = 0.25, y0 = 4.0, ch = 2.15;
  card(s, MX, y0, cw, ch, "1 · Unbiased input",
    "Region intensities are voxel means. A systematic offset in the input is reproduced, not removed — e.g. clamping a CGLS reconstruction passes straight through.", { titleSize: 14 });
  card(s, MX + cw + gap, y0, cw, ch, "2 · Structure \u2265 patch scale",
    "Wedges are piecewise-constant. Geometry finer than the patch, or smooth gradients, cannot be expressed — the fit collapses toward the mean.", { titleSize: 14 });
  card(s, MX + 2 * (cw + gap), y0, cw, ch, "3 · Noise above crossover",
    "The wedge model is biased. It wins only when the noise it averages away costs more than the bias it introduces.", { titleSize: 14 });
  footer(s, 4);
  s.addNotes(
    "Key honest finding: the released defaults never activate the inter-patch consistency terms, so the estimator is just independent wedge fits + averaging. That simple mechanism is what makes the three requirements derivable — and testable."
  );
}

/* ------------- Slide 5 — reproduction ------------- */
{
  const s = pres.addSlide();
  header(s, "Step 0 — reproduce the published benchmark", "Reproduction");
  bullets(s, [
    "Sparse-view CT ablation: engine volume, 20 cone-beam views, P50 Poisson noise.",
    "Pipeline rebuilt end-to-end: R\u00B2-Gaussian benchmark + TIGRE projector + 60 CGLS iterations, then the released FoJ code.",
    "Qualitative result reproduces exactly: quality rises monotonically with patch size R, best at R = 12, with shrinking marginal gains.",
    "Absolute PSNR sits +1.2 to +3.4 dB above reported — consistent with the released M = 2 (vs. M = 3 in the paper) and unspecified reconstruction details.",
  ], MX, 1.7, 5.8, 4.6, { fontSize: 14, gap: 14 });
  s.addText("The implementation behaves as documented — so the domain study uses it as-is.", {
    x: MX, y: 6.35, w: 5.8, h: 0.6, margin: 0,
    fontFace: BODY_FONT, fontSize: 12.5, italic: true, color: SECONDARY,
  });

  const th = { fontFace: BODY_FONT, fontSize: 12, color: "FFFFFF", bold: true, fill: { color: PRIMARY }, align: "center", valign: "middle" };
  const td = { fontFace: BODY_FONT, fontSize: 12, color: TEXT, align: "center", valign: "middle" };
  const tdB = { ...td, bold: true, color: SECONDARY };
  const rows = [
    [{ text: "R", options: th }, { text: "Reported", options: th }, { text: "Ours", options: th }, { text: "\u0394 (dB)", options: th }],
    ["4", "11.33", "14.72", "+3.39"],
    ["6", "14.37", "16.93", "+2.56"],
    ["8", "16.18", "17.99", "+1.81"],
    ["10", "17.17", "18.58", "+1.41"],
    ["12", "17.70", "18.86", "+1.16"],
  ].map((r, i) =>
    i === 0 ? r : r.map((c, j) => ({ text: c, options: j === 3 ? tdB : td }))
  );
  s.addTable(rows, {
    x: 6.9, y: 1.75, w: 5.7, colW: [0.9, 1.6, 1.6, 1.6],
    border: { type: "solid", pt: 0.75, color: "C9D5DE" },
    rowH: 0.42, autoPage: false,
  });
  s.addText("Engine-volume CT ablation — denoised PSNR vs. patch size R", {
    x: 6.9, y: 4.35, w: 5.7, h: 0.35, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED,
  });
  stat(s, 6.9, 4.95, 5.7, "+1.2 \u2192 +3.4 dB", "ours above reported, gap narrows as R grows", { size: 34 });
  footer(s, 5);
  s.addNotes(
    "Before new domains we checked the released code reproduces the paper's headline ablation. It does — same monotone trend, same best patch size. Absolute offset explained by M=2 vs M=3, which the paper itself acknowledges. This validates using the released implementation for the study."
  );
}

/* ------------- Slide 6 — Bentheim (success) ------------- */
{
  const s = pres.addSlide();
  header(s, "Bentheim micro-CT — where FoJ wins", "Result 1 · compatible structure + high noise");
  s.addImage({
    path: `${FIGS}/fig_slices.png`,
    x: (W - 11) / 2, y: 1.45, w: 11, h: 11 * (720 / 3960),
  });
  s.addText("\u03C3 = 55% noise — FoJ keeps grain\u2013pore interfaces sharp while Gaussian, median, and TV blur", {
    x: MX, y: 3.5, w: 12.1, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED,
  });

  const th = { fontFace: BODY_FONT, fontSize: 12, color: "FFFFFF", bold: true, fill: { color: PRIMARY }, align: "center", valign: "middle" };
  const td = { fontFace: BODY_FONT, fontSize: 12, color: TEXT, align: "center", valign: "middle" };
  const tdW = { ...td, bold: true, color: SECONDARY };
  const mk = (sig, noisy, best, foj, fojsWins) => [
    { text: sig, options: td }, { text: noisy, options: td },
    { text: best, options: fojsWins ? td : tdW },
    { text: foj, options: fojsWins ? tdW : td },
  ];
  s.addTable(
    [
      [{ text: "\u03C3", options: th }, { text: "Noisy", options: th }, { text: "Best classical", options: th }, { text: "FoJ", options: th }],
      mk("15%", "16.89", "28.51", "26.55", false),
      mk("30%", "12.00", "26.03", "25.64", false),
      mk("40%", "10.28", "23.39", "24.93", true),
      mk("55%", "8.76", "21.05", "24.29", true),
    ],
    {
      x: MX, y: 4.05, w: 5.9, colW: [1.0, 1.4, 1.9, 1.6],
      border: { type: "solid", pt: 0.75, color: "C9D5DE" },
      rowH: 0.42, autoPage: false,
    }
  );
  s.addText("3D PSNR (dB), 192\u00B3 crop of Bentheim sandstone", {
    x: MX, y: 6.2, w: 5.9, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });

  stat(s, 7.0, 4.0, 5.6, "+3.24 dB", "FoJ over the best classical filter at \u03C3 = 55%", { size: 48 });
  stat(s, 7.0, 5.55, 5.6, "\u2248 11 dB", "input-PSNR crossover — below this, the wedge prior pays off; above it, classical filters win", { size: 30 });
  footer(s, 6);
  s.addNotes(
    "The headline positive result. Classical filters win at 15-30% noise; once input PSNR drops below ~11 dB they either under-smooth or blur interfaces, while FoJ's regional means kill variance and keep planes intact. Boundary-F1 0.85-0.88 vs 0.54 for TV."
  );
}

/* ------------- Slide 7 — MRI (failure 1) ------------- */
{
  const s = pres.addSlide();
  header(s, "BrainWeb MRI — where FoJ fails: smooth texture", "Result 2 · violates piecewise flatness");
  bullets(s, [
    "BrainWeb T1 phantom (181\u00D7217\u00D7181), Rician noise \u03C3 = 3\u201313%.",
    "Tissue gradients are smooth, so constant wedges quantize them into steps — model bias at every noise level.",
    "FoJ trails the 3\u00B3 median filter at all five levels; larger patches only make it worse.",
    "Same story in reconstruction: as a plug-and-play denoiser, FoJ-PnP reaches 27.99 dB vs. 30.39 dB for TV-PnP at 4\u00D7 undersampling.",
  ], MX, 1.7, 6.3, 3.2, { fontSize: 14, gap: 14 });

  const th = { fontFace: BODY_FONT, fontSize: 11.5, color: "FFFFFF", bold: true, fill: { color: PRIMARY }, align: "center", valign: "middle" };
  const td = { fontFace: BODY_FONT, fontSize: 11.5, color: TEXT, align: "center", valign: "middle" };
  const tdB = { ...td, bold: true, color: SECONDARY };
  s.addTable(
    [
      [{ text: "\u03C3", options: th }, { text: "Median", options: th }, { text: "FoJ R=6", options: th }, { text: "gap", options: th }],
      ["3%", "30.80", "25.67", "\u22125.13"],
      ["7%", "24.58", "22.52", "\u22122.06"],
      ["13%", "19.28", "18.31", "\u22120.97"],
    ].map((r, i) =>
      i === 0 ? r : r.map((c, j) => ({ text: c, options: j === 3 ? tdB : td }))
    ),
    {
      x: MX, y: 5.0, w: 5.2, colW: [1.0, 1.4, 1.4, 1.4],
      border: { type: "solid", pt: 0.75, color: "C9D5DE" },
      rowH: 0.34, autoPage: false,
    }
  );
  s.addText("3D PSNR (dB), Rician noise — gap narrows but never closes", {
    x: MX, y: 6.45, w: 5.2, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });

  stat(s, 7.6, 1.9, 5.0, "\u22125.13 dB", "FoJ R=6 vs. median filter at \u03C3 = 3% — below the noisy input itself", { size: 44, color: BADRED });
  stat(s, 7.6, 4.15, 5.0, "0 / 5", "noise levels where FoJ beats any classical baseline on MRI", { size: 44, color: BADRED });
  footer(s, 7);
  s.addNotes(
    "Failure mode one: the wedge prior is piecewise-constant, MRI tissue is smooth — so bias dominates regardless of noise. The gap narrows with noise (variance reduction helps) but never closes. Plug-and-play reconstruction confirms it's the prior, not the task."
  );
}

/* ------------- Slide 8 — binary media (failure 2) ------------- */
{
  const s = pres.addSlide();
  header(s, "Binary porous media — where FoJ fails: sub-patch features", "Result 3 · violates structure scale");
  s.addImage({
    path: `${FIGS}/fig_collapse.png`,
    x: (W - 9) / 2, y: 1.6, w: 9, h: 9 * (720 / 2160),
  });
  s.addText("Berea, \u03C3 = 15%: binary ground truth (left) \u2192 FoJ output (right) collapses toward the volume mean", {
    x: MX, y: 4.7, w: 12.1, h: 0.3, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10.5, italic: true, color: MUTED,
  });
  const cw = 3.87, gap = 0.25, y0 = 5.15, ch = 1.75;
  card(s, MX, y0, cw, ch, "19.5 \u2192 7.1 dB",
    "FoJ scores below the noisy input — denoising destroys the signal it should keep.", { titleSize: 16, bodySize: 12 });
  card(s, MX + cw + gap, y0, cw, ch, "\u03C3 0.49 \u2192 0.09",
    "Output intensity spread shrinks 5\u00D7: the volume comes out nearly flat.", { titleSize: 16, bodySize: 12 });
  card(s, MX + 2 * (cw + gap), y0, cw, ch, "Why",
    "Features are ~2 voxels; three planes through one vertex can't isolate dozens of pores in an R\u00B3 patch, so every wedge mean \u2248 global mean.", { titleSize: 16, bodySize: 12 });
  footer(s, 8);
  s.addNotes(
    "Failure mode two, the most dramatic: geometry finer than the patch can't be partitioned by three planes, so wedge means degenerate to the global mean — output nearly flat. Even perfect structure type (binary = piecewise-constant) fails when it's below patch scale."
  );
}

/* ------------- Slide 9 — unified story ------------- */
{
  const s = pres.addSlide();
  header(s, "One mechanism explains all three outcomes", "Synthesis");
  s.addImage({
    path: `${FIGS}/fig_domains.png`,
    x: 7.15, y: 1.75, w: 5.55, h: 5.55 * (1020 / 1650),
  });
  s.addText("FoJ margin (PSNR \u2212 best classical) vs. input PSNR across domains", {
    x: 7.15, y: 5.25, w: 5.55, h: 0.35, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 10, italic: true, color: MUTED,
  });
  s.addText("FoJ helps only when all three hold:", {
    x: MX, y: 1.75, w: 6.2, h: 0.35, margin: 0,
    fontFace: BODY_FONT, fontSize: 15, bold: true, color: PRIMARY,
  });
  bullets(s, [
    "unbiased input (means reproduce offsets),",
    "piecewise-flat structure at \u2265 patch scale,",
    "noise high enough to pay for the wedge bias.",
  ], MX + 0.15, 2.2, 6.2, 1.5, { fontSize: 13.5, gap: 7 });
  s.addText(
    [
      { text: "\u2713 Bentheim micro-CT", options: { bold: true, color: OKGREEN, breakLine: true } },
      { text: "   piecewise-flat grains > patch + input < 11 dB \u2192 best method (+3.2 dB)", options: { color: TEXT, breakLine: true, paraSpaceAfter: 12 } },
      { text: "\u2717 BrainWeb MRI", options: { bold: true, color: BADRED, breakLine: true } },
      { text: "   smooth gradients violate flatness \u2192 loses at every noise level", options: { color: TEXT, breakLine: true, paraSpaceAfter: 12 } },
      { text: "\u2717 Binary porous media", options: { bold: true, color: BADRED, breakLine: true } },
      { text: "   ~2-voxel features < patch \u2192 collapses to the mean", options: { color: TEXT } },
    ],
    { x: MX, y: 3.95, w: 6.2, h: 2.8, margin: 0, fontFace: BODY_FONT, fontSize: 13.5, valign: "top" }
  );
  footer(s, 9);
  s.addNotes(
    "The synthesis slide — the plot shows the same crossover on every domain: margin flips positive around ~11 dB input, but ONLY where structure is expressible. Structure decides whether FoJ can win; noise decides when it does."
  );
}

/* ------------- Slide 10 — takeaways (dark) ------------- */
{
  const s = pres.addSlide();
  s.background = { color: DARK };
  s.addText("Takeaways", {
    x: 0.8, y: 0.8, w: W - 1.6, h: 0.8, margin: 0, align: "center",
    fontFace: TITLE_FONT, fontSize: 34, bold: true, color: "FFFFFF",
  });
  s.addText("3D FoJ is a structural prior, not a universal denoiser.", {
    x: 0.8, y: 1.85, w: W - 1.6, h: 0.5, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 18, italic: true, color: ACCENT,
  });
  const lines = [
    ["It wins", "where structure is piecewise-flat at patch scale and noise is severe — Bentheim micro-CT at 8.8 dB input: +3.2 dB over every classical filter."],
    ["It loses", "where gradients are smooth (MRI) or features are finer than a patch (binary media) — classical filters are better and cheaper."],
    ["Before adopting it", "check three things: is the input unbiased, is the structure expressible at patch scale, and is the noise above the crossover."],
  ];
  lines.forEach(([lead, rest], i) => {
    const y = 2.75 + i * 1.05;
    s.addText(
      [
        { text: lead + " ", options: { bold: true, color: ACCENT } },
        { text: rest, options: { color: LIGHT_ON_DARK } },
      ],
      { x: 1.5, y, w: W - 3, h: 0.95, margin: 0, fontFace: BODY_FONT, fontSize: 15, valign: "top" }
    );
  });
  s.addText("github.com/samudraneel05/SectionC_Team17", {
    x: 0.8, y: 6.15, w: W - 1.6, h: 0.4, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 16, bold: true, color: "FFFFFF",
  });
  s.addText("Thank you — questions?", {
    x: 0.8, y: 6.6, w: W - 1.6, h: 0.35, margin: 0, align: "center",
    fontFace: BODY_FONT, fontSize: 13, color: "8FA8C8",
  });
  s.addNotes(
    "Close on the honest framing: FoJ isn't better or worse overall — it has knowable conditions. The three checks let a practitioner predict the outcome before running it."
  );
}

const out = process.argv[2] || "presentation.pptx";
pres.writeFile({ fileName: out }).then(() => console.log("wrote", out));
