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
    const offset = p5.Vector.fromAngle(canonical + jitter).mult(radius);
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
    this.vel = p5.Vector.random2D().mult(random(0.2, 1.1));
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
    this.vel = p5.Vector.fromAngle(angle + HALF_PI * random([-1, 1])).mult(random(0.2, 0.9));
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
  const toCenter = p5.Vector.sub(center, pos);
  const dCenter = max(1, toCenter.mag());
  const field = createVector(0, 0);

  const scale = 0.0018 + params.turbulence * 0.0027;
  const n1 = noise(pos.x * scale, pos.y * scale, fieldSalt + systemAge * 0.0016);
  const n2 = noise(pos.y * scale * 1.7 + 31.4, pos.x * scale * 1.7, fieldSalt * 0.37);
  const angle = (n1 * TWO_PI * 2.8) + (n2 - 0.5) * PI * params.turbulence + sin(memory) * 0.16;
  field.add(p5.Vector.fromAngle(angle).mult(0.52 * params.turbulence));

  for (const clause of clauses) {
    const delta = p5.Vector.sub(clause.pos, pos);
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

  const symmetry = p5.Vector.sub(createVector(width - pos.x, height - pos.y), pos).normalize().mult(0.045 * params.symmetryPressure);
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
