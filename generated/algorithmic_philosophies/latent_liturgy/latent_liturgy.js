/*
 * First-party Canvas runtime for Latent Liturgy.
 *
 * This intentionally implements only the small drawing/vector surface this
 * artwork needs. Keeping the viewer self-contained avoids shipping a separate
 * JavaScript framework inside the signed iOS application.
 */
'use strict';

const PI = Math.PI;
const HALF_PI = Math.PI / 2;
const TWO_PI = Math.PI * 2;
const RGB = 'rgb';
const BLEND = 'source-over';
const ADD = 'lighter';

const floor = Math.floor;
const sqrt = Math.sqrt;
const cos = Math.cos;
const sin = Math.sin;
const min = Math.min;
const max = Math.max;
const exp = Math.exp;
const sq = value => value * value;

let width = 0;
let height = 0;
let mainCanvas;
let mainSurface;
let animationFrameID;
let animationRunning = false;
let randomState = 1;
let noiseState = 1;

function clamp(value, lower, upper) {
  return Math.min(upper, Math.max(lower, value));
}

function map(value, inputMin, inputMax, outputMin, outputMax, withinBounds = false) {
  const inputSpan = inputMax - inputMin;
  const ratio = inputSpan === 0 ? 0 : (value - inputMin) / inputSpan;
  const mapped = outputMin + ratio * (outputMax - outputMin);
  if (!withinBounds) return mapped;
  return outputMin < outputMax
    ? clamp(mapped, outputMin, outputMax)
    : clamp(mapped, outputMax, outputMin);
}

function randomSeed(seed) {
  randomState = (Number(seed) >>> 0) || 1;
}

function nextRandom() {
  randomState = (randomState + 0x6d2b79f5) >>> 0;
  let value = randomState;
  value = Math.imul(value ^ (value >>> 15), value | 1);
  value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
  return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
}

function random(lowerOrValues, upper) {
  const unit = nextRandom();
  if (Array.isArray(lowerOrValues)) {
    return lowerOrValues[Math.min(lowerOrValues.length - 1, Math.floor(unit * lowerOrValues.length))];
  }
  if (lowerOrValues === undefined) return unit;
  if (upper === undefined) return unit * lowerOrValues;
  return lowerOrValues + unit * (upper - lowerOrValues);
}

function noiseSeed(seed) {
  noiseState = (Number(seed) >>> 0) || 1;
}

function smoothstep(value) {
  return value * value * (3 - 2 * value);
}

function latticeNoise(x, y, z) {
  let hash = noiseState;
  hash ^= Math.imul(x, 0x27d4eb2d);
  hash ^= Math.imul(y, 0x165667b1);
  hash ^= Math.imul(z, 0x1b873593);
  hash = Math.imul(hash ^ (hash >>> 15), 0x85ebca6b);
  hash = Math.imul(hash ^ (hash >>> 13), 0xc2b2ae35);
  return ((hash ^ (hash >>> 16)) >>> 0) / 4294967295;
}

function noise(x, y = 0, z = 0) {
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const tx = smoothstep(x - x0);
  const ty = smoothstep(y - y0);
  const tz = smoothstep(z - z0);
  const mix = (a, b, amount) => a + (b - a) * amount;
  const zPlane = dz => {
    const top = mix(latticeNoise(x0, y0, z0 + dz), latticeNoise(x0 + 1, y0, z0 + dz), tx);
    const bottom = mix(latticeNoise(x0, y0 + 1, z0 + dz), latticeNoise(x0 + 1, y0 + 1, z0 + dz), tx);
    return mix(top, bottom, ty);
  };
  return mix(zPlane(0), zPlane(1), tz);
}

class Vector {
  constructor(x = 0, y = 0) {
    this.x = x;
    this.y = y;
  }

  copy() {
    return new Vector(this.x, this.y);
  }

  set(xOrVector, y) {
    if (xOrVector instanceof Vector) {
      this.x = xOrVector.x;
      this.y = xOrVector.y;
    } else {
      this.x = xOrVector;
      this.y = y;
    }
    return this;
  }

  add(vector) {
    this.x += vector.x;
    this.y += vector.y;
    return this;
  }

  mult(value) {
    this.x *= value;
    this.y *= value;
    return this;
  }

  mag() {
    return Math.hypot(this.x, this.y);
  }

  normalize() {
    const magnitude = this.mag();
    if (magnitude > 0) this.mult(1 / magnitude);
    return this;
  }

  limit(maximum) {
    const magnitude = this.mag();
    if (magnitude > maximum && magnitude > 0) this.mult(maximum / magnitude);
    return this;
  }

  static fromAngle(angle) {
    return new Vector(Math.cos(angle), Math.sin(angle));
  }

  static random2D() {
    return Vector.fromAngle(random(TWO_PI));
  }

  static sub(left, right) {
    return new Vector(left.x - right.x, left.y - right.y);
  }
}

function createVector(x, y) {
  return new Vector(x, y);
}

function parseColor(value) {
  if (typeof value === 'object' && value !== null && 'r' in value) return value;
  if (typeof value !== 'string') return { r: Number(value) || 0, g: 0, b: 0, a: 255 };
  const match = value.trim().match(/^#([0-9a-f]{3}|[0-9a-f]{6})$/i);
  if (!match) return { r: 0, g: 0, b: 0, a: 255 };
  const hex = match[1].length === 3
    ? match[1].split('').map(character => character + character).join('')
    : match[1];
  return {
    r: parseInt(hex.slice(0, 2), 16),
    g: parseInt(hex.slice(2, 4), 16),
    b: parseInt(hex.slice(4, 6), 16),
    a: 255
  };
}

function color(value) {
  return parseColor(value);
}

function red(value) { return parseColor(value).r; }
function green(value) { return parseColor(value).g; }
function blue(value) { return parseColor(value).b; }

function lerpColor(start, end, amount) {
  const left = parseColor(start);
  const right = parseColor(end);
  const fraction = clamp(amount, 0, 1);
  return {
    r: left.r + (right.r - left.r) * fraction,
    g: left.g + (right.g - left.g) * fraction,
    b: left.b + (right.b - left.b) * fraction,
    a: left.a + (right.a - left.a) * fraction
  };
}

function cssColor(value, alpha) {
  const parsed = parseColor(value);
  const resolvedAlpha = alpha === undefined ? parsed.a : alpha;
  return `rgba(${Math.round(parsed.r)}, ${Math.round(parsed.g)}, ${Math.round(parsed.b)}, ${clamp(resolvedAlpha / 255, 0, 1)})`;
}

class CanvasSurface {
  constructor(canvas) {
    this.canvas = canvas;
    this.context = canvas.getContext('2d', { alpha: true });
    this.fillEnabled = true;
    this.strokeEnabled = true;
    this.stateStack = [];
  }

  pixelDensity() { }
  colorMode() { }

  push() {
    this.stateStack.push({ fillEnabled: this.fillEnabled, strokeEnabled: this.strokeEnabled });
    this.context.save();
  }

  pop() {
    const state = this.stateStack.pop();
    this.context.restore();
    if (state) {
      this.fillEnabled = state.fillEnabled;
      this.strokeEnabled = state.strokeEnabled;
    }
  }

  blendMode(mode) {
    this.context.globalCompositeOperation = mode;
  }

  noFill() {
    this.fillEnabled = false;
  }

  noStroke() {
    this.strokeEnabled = false;
  }

  fill(redOrColor, greenValue, blueValue, alpha = 255) {
    this.fillEnabled = true;
    this.context.fillStyle = typeof redOrColor === 'string' || typeof redOrColor === 'object'
      ? cssColor(redOrColor, greenValue)
      : cssColor({ r: redOrColor, g: greenValue, b: blueValue, a: alpha });
  }

  stroke(redOrColor, greenValue, blueValue, alpha = 255) {
    this.strokeEnabled = true;
    this.context.strokeStyle = typeof redOrColor === 'string' || typeof redOrColor === 'object'
      ? cssColor(redOrColor, greenValue)
      : cssColor({ r: redOrColor, g: greenValue, b: blueValue, a: alpha });
  }

  strokeWeight(value) {
    this.context.lineWidth = value;
  }

  background(value) {
    this.context.save();
    this.context.globalCompositeOperation = BLEND;
    this.context.fillStyle = cssColor(value);
    this.context.fillRect(0, 0, this.canvas.width, this.canvas.height);
    this.context.restore();
  }

  paintPath() {
    if (this.fillEnabled) this.context.fill();
    if (this.strokeEnabled) this.context.stroke();
  }

  rect(x, y, rectWidth, rectHeight, radius = 0) {
    const context = this.context;
    const boundedRadius = clamp(radius, 0, Math.min(Math.abs(rectWidth), Math.abs(rectHeight)) / 2);
    context.beginPath();
    if (boundedRadius === 0) {
      context.rect(x, y, rectWidth, rectHeight);
    } else {
      const right = x + rectWidth;
      const bottom = y + rectHeight;
      context.moveTo(x + boundedRadius, y);
      context.lineTo(right - boundedRadius, y);
      context.quadraticCurveTo(right, y, right, y + boundedRadius);
      context.lineTo(right, bottom - boundedRadius);
      context.quadraticCurveTo(right, bottom, right - boundedRadius, bottom);
      context.lineTo(x + boundedRadius, bottom);
      context.quadraticCurveTo(x, bottom, x, bottom - boundedRadius);
      context.lineTo(x, y + boundedRadius);
      context.quadraticCurveTo(x, y, x + boundedRadius, y);
    }
    this.paintPath();
  }

  circle(x, y, diameter) {
    this.context.beginPath();
    this.context.arc(x, y, diameter / 2, 0, TWO_PI);
    this.paintPath();
  }

  line(x1, y1, x2, y2) {
    if (!this.strokeEnabled) return;
    this.context.beginPath();
    this.context.moveTo(x1, y1);
    this.context.lineTo(x2, y2);
    this.context.stroke();
  }
}

function createCanvas(canvasWidth, canvasHeight) {
  width = canvasWidth;
  height = canvasHeight;
  mainCanvas = document.createElement('canvas');
  mainCanvas.width = canvasWidth;
  mainCanvas.height = canvasHeight;
  mainCanvas.setAttribute('role', 'img');
  mainCanvas.setAttribute('aria-label', 'Latent Liturgy generative particle artwork');
  mainSurface = new CanvasSurface(mainCanvas);
  return {
    parent(elementID) {
      document.getElementById(elementID).replaceChildren(mainCanvas);
    }
  };
}

function createGraphics(canvasWidth, canvasHeight) {
  const canvas = document.createElement('canvas');
  canvas.width = canvasWidth;
  canvas.height = canvasHeight;
  return new CanvasSurface(canvas);
}

function pixelDensity() { }
function background(value) { mainSurface.background(value); }
function push() { mainSurface.push(); }
function pop() { mainSurface.pop(); }
function blendMode(mode) { mainSurface.blendMode(mode); }
function noFill() { mainSurface.noFill(); }
function noStroke() { mainSurface.noStroke(); }
function fill(...values) { mainSurface.fill(...values); }
function stroke(...values) { mainSurface.stroke(...values); }
function strokeWeight(value) { mainSurface.strokeWeight(value); }
function circle(x, y, diameter) { mainSurface.circle(x, y, diameter); }
function rect(x, y, rectWidth, rectHeight, radius) { mainSurface.rect(x, y, rectWidth, rectHeight, radius); }

function image(surface, x, y) {
  mainSurface.context.save();
  mainSurface.context.globalCompositeOperation = BLEND;
  mainSurface.context.drawImage(surface.canvas, x, y);
  mainSurface.context.restore();
}

function renderFrame() {
  animationFrameID = undefined;
  if (!animationRunning) return;
  draw();
  if (animationRunning) animationFrameID = requestAnimationFrame(renderFrame);
}

function loop() {
  animationRunning = true;
  if (animationFrameID === undefined) animationFrameID = requestAnimationFrame(renderFrame);
}

function noLoop() {
  animationRunning = false;
  if (animationFrameID !== undefined) {
    cancelAnimationFrame(animationFrameID);
    animationFrameID = undefined;
  }
}

const defaultParams = {
  seed: 12345,
  clauseCount: 7,
  particleCount: 850,
  turbulence: 0.78,
  clauseGravity: 1.18,
  processionSpeed: 1.35,
  trailPatience: 38,
  symmetryPressure: 0.42,
  marginBend: 0.62,
  backgroundColor: '#11110f',
  inkColor: '#f4e7cf',
  emberColor: '#d97757',
  witnessColor: '#6a9bcc'
};

let params = { ...defaultParams };
let particles = [];
let clauses = [];
let trailLayer;
let systemAge = 0;
let maxAge = 980;
let fieldSalt = 0;

function setup() {
  const canvas = createCanvas(1200, 1200);
  canvas.parent('canvas-container');
  pixelDensity(1);
  initializeSystem();
}

function initializeSystem() {
  params.seed = Number.isFinite(params.seed) ? params.seed : defaultParams.seed;
  randomSeed(params.seed);
  noiseSeed(params.seed);
  fieldSalt = random(1000);
  systemAge = 0;
  maxAge = floor(map(params.trailPatience, 10, 80, 620, 1300));

  trailLayer = createGraphics(width, height);
  trailLayer.pixelDensity(1);
  trailLayer.colorMode(RGB, 255, 255, 255, 255);
  trailLayer.background(params.backgroundColor);
  trailLayer.noFill();
  trailLayer.blendMode(BLEND);

  clauses = buildClauses();
  particles = buildParticles();
  background(params.backgroundColor);
  loop();
}

function buildClauses() {
  const built = [];
  const count = floor(params.clauseCount);
  const golden = PI * (3 - sqrt(5));
  const center = createVector(width * 0.5, height * 0.5);
  for (let i = 0; i < count; i++) {
    const canonical = -HALF_PI + i * golden;
    const jitter = random(-0.24, 0.24) * (1 - params.symmetryPressure);
    const radius = width * (0.105 + 0.245 * sqrt((i + 0.5) / count));
    const offset = Vector.fromAngle(canonical + jitter).mult(radius);
    const vow = createVector(center.x + offset.x, center.y + offset.y);
    built.push({
      pos: vow,
      phase: random(TWO_PI),
      polarity: random([-1, 1]),
      radius: random(width * 0.12, width * 0.26),
      gravity: random(0.72, 1.28)
    });
  }
  return built;
}

function buildParticles() {
  const built = [];
  const count = floor(params.particleCount);
  for (let i = 0; i < count; i++) {
    let x;
    let y;
    if (random() < 0.68) {
      const clause = random(clauses);
      const angle = random(TWO_PI);
      const radius = random(width * 0.035, width * 0.34) * sqrt(random());
      x = clause.pos.x + cos(angle) * radius;
      y = clause.pos.y + sin(angle) * radius;
    } else {
      const side = floor(random(4));
      if (side === 0) { x = random(width); y = -random(20, 120); }
      else if (side === 1) { x = width + random(20, 120); y = random(height); }
      else if (side === 2) { x = random(width); y = height + random(20, 120); }
      else { x = -random(20, 120); y = random(height); }
    }

    built.push(new Particle(x, y, i));
  }
  return built;
}

class Particle {
  constructor(x, y, index) {
    this.pos = createVector(x, y);
    this.prev = this.pos.copy();
    this.vel = Vector.random2D().mult(random(0.2, 1.1));
    this.index = index;
    this.life = random(0.55, 1.0);
    this.weight = random(0.28, 1.24);
    this.hueBias = random();
    this.memory = random(TWO_PI);
  }

  step() {
    this.prev.set(this.pos);
    const force = sampleField(this.pos, this.index, this.memory);
    this.vel.mult(0.936);
    this.vel.add(force.mult(params.processionSpeed));
    this.vel.limit(2.9 + params.processionSpeed * 1.35);
    this.pos.add(this.vel);
    this.memory += 0.006 + this.vel.mag() * 0.002;
    this.life -= 0.00024;

    if (this.isExpired()) {
      this.reseed();
    }
  }

  isExpired() {
    return this.life <= 0 || this.pos.x < -140 || this.pos.x > width + 140 || this.pos.y < -140 || this.pos.y > height + 140;
  }

  reseed() {
    const angle = random(TWO_PI);
    const radius = random(width * 0.48, width * 0.64);
    this.pos.set(width / 2 + cos(angle) * radius, height / 2 + sin(angle) * radius);
    this.prev.set(this.pos);
    this.vel = Vector.fromAngle(angle + HALF_PI * random([-1, 1])).mult(random(0.2, 0.9));
    this.life = random(0.58, 1.0);
    this.memory = random(TWO_PI);
  }

  drawTo(layer) {
    const speed = this.vel.mag();
    const warm = color(params.emberColor);
    const pale = color(params.inkColor);
    const witness = color(params.witnessColor);
    const phase = 0.5 + 0.5 * sin(this.memory + this.index * 0.013);
    const c = this.hueBias > 0.82 ? lerpColor(pale, witness, phase * 0.65) : lerpColor(pale, warm, min(1, speed / 4.2));
    const alpha = map(speed, 0, 4.5, 14, 58, true) * this.life;

    layer.stroke(red(c), green(c), blue(c), alpha);
    layer.strokeWeight(this.weight * map(speed, 0, 4, 0.42, 1.7, true));
    layer.line(this.prev.x, this.prev.y, this.pos.x, this.pos.y);
  }
}

function sampleField(pos, index, memory) {
  const center = createVector(width * 0.5, height * 0.5);
  const toCenter = Vector.sub(center, pos);
  const dCenter = max(1, toCenter.mag());
  const field = createVector(0, 0);

  const scale = 0.0018 + params.turbulence * 0.0027;
  const n1 = noise(pos.x * scale, pos.y * scale, fieldSalt + systemAge * 0.0016);
  const n2 = noise(pos.y * scale * 1.7 + 31.4, pos.x * scale * 1.7, fieldSalt * 0.37);
  const angle = (n1 * TWO_PI * 2.8) + (n2 - 0.5) * PI * params.turbulence + sin(memory) * 0.16;
  field.add(Vector.fromAngle(angle).mult(0.52 * params.turbulence));

  for (const clause of clauses) {
    const delta = Vector.sub(clause.pos, pos);
    const d = max(8, delta.mag());
    const influence = exp(-sq(d / clause.radius)) * params.clauseGravity * clause.gravity;
    const orbital = createVector(-delta.y, delta.x).normalize().mult(influence * clause.polarity * 0.92);
    const pull = delta.normalize().mult(influence * 0.34);
    field.add(orbital).add(pull);
  }

  const margin = min(pos.x, pos.y, width - pos.x, height - pos.y);
  if (margin < width * 0.18) {
    const tangent = createVector(-toCenter.y, toCenter.x).normalize();
    const pressure = map(margin, 0, width * 0.18, 1, 0, true);
    field.add(tangent.mult(pressure * params.marginBend * 0.94));
  }

  const symmetry = Vector.sub(createVector(width - pos.x, height - pos.y), pos).normalize().mult(0.045 * params.symmetryPressure);
  const breath = sin(systemAge * 0.012 + index * 0.037) * 0.028;
  field.add(symmetry);
  field.add(toCenter.normalize().mult((0.18 + breath) / sqrt(dCenter / 120)));
  return field;
}

function draw() {
  if (!trailLayer) return;

  trailLayer.noStroke();
  const fadeAlpha = map(params.trailPatience, 10, 80, 6.5, 0.75);
  const bg = color(params.backgroundColor);
  trailLayer.fill(red(bg), green(bg), blue(bg), fadeAlpha);
  trailLayer.rect(0, 0, width, height);

  const steps = params.particleCount > 1100 ? 1 : 2;
  for (let s = 0; s < steps; s++) {
    for (const p of particles) {
      p.step();
      p.drawTo(trailLayer);
    }
    systemAge++;
  }

  image(trailLayer, 0, 0);
  drawClauseGlints();
  drawVignette();

  if (systemAge > maxAge) {
    noLoop();
  }
}

function drawClauseGlints() {
  push();
  blendMode(ADD);
  noFill();
  for (const clause of clauses) {
    const pulse = 0.5 + 0.5 * sin(systemAge * 0.018 + clause.phase);
    const c = color(params.emberColor);
    stroke(red(c), green(c), blue(c), 18 + pulse * 28);
    strokeWeight(0.7 + pulse * 0.8);
    circle(clause.pos.x, clause.pos.y, clause.radius * (0.12 + pulse * 0.035));
  }
  pop();
}

function drawVignette() {
  push();
  noFill();
  for (let i = 0; i < 42; i++) {
    const alpha = map(i, 0, 41, 0, 8);
    stroke(0, 0, 0, alpha);
    strokeWeight(8);
    rect(i * 6, i * 6, width - i * 12, height - i * 12, 18);
  }
  pop();
}

function regenerateArtwork() {
  initializeSystem();
}

function updateSeed() {
  params.seed = parseInt(document.getElementById('seed-input').value, 10);
  initializeSystem();
}

function previousSeed() {
  params.seed--;
  document.getElementById('seed-input').value = params.seed;
  initializeSystem();
}

function nextSeed() {
  params.seed++;
  document.getElementById('seed-input').value = params.seed;
  initializeSystem();
}

function randomSeedAndUpdate() {
  params.seed = Math.floor(Math.random() * 999999);
  document.getElementById('seed-input').value = params.seed;
  initializeSystem();
}

function resetParameters() {
  const currentSeed = params.seed;
  params = { ...defaultParams, seed: currentSeed };
  syncControls();
  initializeSystem();
}

function updateParam(name, value) {
  params[name] = parseFloat(value);
  document.getElementById(name + '-value').textContent = value;
  initializeSystem();
}

function updateColor(id, value) {
  params[id] = value;
  initializeSystem();
}

function syncControls() {
  document.getElementById('seed-input').value = params.seed;
  for (const key of ['clauseCount', 'particleCount', 'turbulence', 'clauseGravity', 'processionSpeed', 'trailPatience', 'symmetryPressure', 'marginBend']) {
    const input = document.getElementById(key);
    const readout = document.getElementById(key + '-value');
    if (input) input.value = params[key];
    if (readout) readout.textContent = params[key];
  }
  for (const key of ['backgroundColor', 'inkColor', 'emberColor', 'witnessColor']) {
    const input = document.getElementById(key);
    if (input) input.value = params[key];
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', setup, { once: true });
} else {
  setup();
}
