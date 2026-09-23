'use strict';

// This page explains the harness. It never invokes it or dispatches GPU work.
const byId = id => document.getElementById(id);
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
const machine = byId('machine');
const scene = byId('scene');
let tilt = 58;
let rotation = -35;
let drag = null;
const drawModel = () => {
  machine.style.transform = `translate(-50%,-50%) rotateX(${tilt}deg) rotateZ(${rotation}deg)`;
};
scene.addEventListener('pointerdown', event => {
  if (event.button !== 0) return;
  drag = {x: event.clientX, y: event.clientY, tilt, rotation};
  scene.setPointerCapture(event.pointerId);
});
scene.addEventListener('pointermove', event => {
  if (!drag) return;
  rotation = drag.rotation + (event.clientX - drag.x) * .35;
  tilt = Math.max(25, Math.min(75, drag.tilt - (event.clientY - drag.y) * .2));
  drawModel();
});
['pointerup', 'pointercancel', 'lostpointercapture'].forEach(name => scene.addEventListener(name, () => {drag = null;}));
scene.addEventListener('keydown', event => {
  if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
  event.preventDefault();
  if (event.key === 'ArrowLeft') rotation -= 5;
  if (event.key === 'ArrowRight') rotation += 5;
  if (event.key === 'ArrowUp') tilt = Math.min(75, tilt + 5);
  if (event.key === 'ArrowDown') tilt = Math.max(25, tilt - 5);
  drawModel();
});
byId('separation').addEventListener('input', event => machine.style.setProperty('--spread', event.target.value));
byId('reset-view').addEventListener('click', () => {
  tilt = 58; rotation = -35;
  byId('separation').value = 58;
  machine.style.setProperty('--spread', 58);
  drawModel();
});

const stages = [
  ['Propose', 'Give the model a small job.', 'The model receives the current engine, profile, recent log and allowed playbook. It returns complete source files with a hypothesis and expected effect.', 'proposal\n  item\n  hypothesis\n  expected_effect\n  files\n  risk\n  reasoning'],
  ['Guard', 'Check the source before running it.', 'Validate file paths, syntax and allowed operations. The proposal can edit the engine and its kernels. It cannot replace the judge or its tests.', 'allowed files\n  engine/engine.py\n  engine/kernels/*.py\n\nsource guard\n  parse\n  inspect\n  accept or reject'],
  ['Test', 'Find the broken assumption.', 'Run unit tests on L4. On failure, send the error back to the same model for up to two repairs. Then check public workloads against native replay.', 'unit tests\n  fail → repair (up to 2)\n  pass → public check\n\npublic replay\n  fail → restore\n  pass → benchmark'],
  ['Measure', 'Time the complete generation.', 'Run three H100 samples per selected workload. Measure first-token latency, decode latency, throughput, memory and spread against a matching native baseline.', 'H100 / 3 samples\n  first token\n  remaining tokens\n  total throughput\n  memory\n  spread'],
  ['Decide', 'Let the measurements decide.', 'A keep needs all gates to pass, over 1 percent aggregate improvement and no workload regression above 3 percent. Save the candidate and outcome either way.', 'all gates pass\n  + improvement > 1%\n  + regression ≤ 3%\n    → keep locally\n\notherwise\n    → snapshot and restore']
];
let stage = 0;
let timer = null;
const stageButtons = [...document.querySelectorAll('[data-stage]')];
function showStage(index) {
  stage = index;
  const [name, title, description, code] = stages[index];
  byId('stage-tag').textContent = `${String(index + 1).padStart(2, '0')} / ${name.toUpperCase()}`;
  byId('stage-title').textContent = title;
  byId('stage-description').textContent = description;
  byId('stage-code').textContent = code;
  stageButtons.forEach((button, i) => button.setAttribute('aria-pressed', String(i === index)));
}
function stopCycle() {
  clearInterval(timer); timer = null;
  byId('play-loop').textContent = reducedMotion.matches ? 'Next stage →' : 'Play cycle ▷';
  byId('play-loop').setAttribute('aria-pressed', 'false');
}
stageButtons.forEach(button => button.addEventListener('click', () => {stopCycle(); showStage(Number(button.dataset.stage));}));
byId('play-loop').addEventListener('click', () => {
  if (reducedMotion.matches) {showStage((stage + 1) % stages.length); return;}
  if (timer) {stopCycle(); return;}
  showStage((stage + 1) % stages.length);
  timer = setInterval(() => showStage((stage + 1) % stages.length), 3200);
  byId('play-loop').textContent = 'Pause cycle Ⅱ';
  byId('play-loop').setAttribute('aria-pressed', 'true');
});
reducedMotion.addEventListener('change', stopCycle);
document.addEventListener('visibilitychange', () => {if (document.hidden) stopCycle();});
new IntersectionObserver(entries => {if (!entries[0].isIntersecting) stopCycle();}).observe(byId('loop'));
stopCycle();

byId('logit-gap').addEventListener('input', event => {
  const gap = Number(event.target.value);
  const passes = gap <= 2;
  byId('gap-value').textContent = gap.toFixed(1);
  byId('candidate-logit').textContent = (10 - gap).toFixed(1);
  byId('candidate-bar').style.width = `${(10 - gap) * 10}%`;
  byId('candidate-bar').style.background = passes ? 'var(--green)' : 'var(--amber)';
  const verdict = byId('verdict');
  verdict.classList.toggle('fail', !passes);
  verdict.replaceChildren(document.createTextNode(passes ? 'Pass ' : 'Fail '));
  const comparison = document.createElement('span');
  comparison.textContent = `${gap.toFixed(1)} ${passes ? '≤' : '>'} 2.0`;
  verdict.append(comparison);
});

document.querySelectorAll('.copy-button').forEach(button => button.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(byId(button.dataset.copy).textContent.trim());
    button.textContent = 'Copied';
  } catch {
    button.textContent = 'Select text to copy';
  }
  setTimeout(() => {button.textContent = 'Copy';}, 2500);
}));
const links = [...document.querySelectorAll('.contents nav a')];
const sectionObserver = new IntersectionObserver(entries => {
  const current = entries.filter(entry => entry.isIntersecting).sort((a,b) => a.boundingClientRect.top - b.boundingClientRect.top)[0];
  if (!current) return;
  links.forEach(link => {
    const active = link.hash === `#${current.target.id}`;
    link.classList.toggle('active', active);
    if (active) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current');
  });
}, {rootMargin: '-10% 0px -65% 0px'});
document.querySelectorAll('article > section').forEach(section => sectionObserver.observe(section));
