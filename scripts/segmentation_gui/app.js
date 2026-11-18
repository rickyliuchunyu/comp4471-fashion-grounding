'use strict';
const $ = (id) => document.getElementById(id);
const colors = [
  '',
  '#26b990',
  '#677ce1',
  '#f18661',
  '#e3a542',
  '#d676b1',
  '#e5ba37',
  '#4daac7',
  '#b87be1'
];
const state = {
  mode: 'gallery',
  model: null,
  index: 0,
  ids: [],
  categories: {},
  selected: null,
  result: null,
  version: 0,
  file: null,
  preview: null,
  uploaded: null,
  busy: false,
  token: ''
};

function message(text) {
  $('message').textContent = text;
}
function loadImage(url) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('图片或掩膜加载失败。'));
    image.src = url;
  });
}
async function jsonResponse(response) {
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error('服务连接中断，请重新打开工作台。');
  }
  if (!response.ok) throw new Error(data.error || '读取结果失败。');
  return data;
}
function clear(text) {
  state.result = null;
  state.selected = null;
  $('visual').hidden = true;
  $('empty').hidden = false;
  $('empty').textContent = text;
  $('list').replaceChildren();
  $('count').textContent = '0';
  $('reset').hidden = true;
  $('dimensions').textContent = '—';
  message('');
}
function visibleInstances() {
  return state.result.instances.filter((x) => x.score >= Number($('threshold').value) / 100);
}
function draw() {
  if (!state.result) return;
  const canvas = $('canvas'),
    ctx = canvas.getContext('2d');
  const { size, original } = state.result;
  if (canvas.width !== size[0] || canvas.height !== size[1]) [canvas.width, canvas.height] = size;
  ctx.globalAlpha = 1;
  ctx.drawImage(original, 0, 0, ...size);
  const instances = visibleInstances();
  if (!instances.some((x) => x.query === state.selected)) state.selected = null;
  const focus =
    state.selected === null ? instances : instances.filter((x) => x.query === state.selected);
  for (const inst of [...focus].reverse()) {
    const [x, y, right, bottom] = inst.box;
    ctx.globalAlpha = state.selected === null ? 0.54 : 0.72;
    ctx.drawImage(inst.layer, x, y);
    if ($('boxes').checked) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = colors[inst.category];
      ctx.lineWidth = Math.max(...size) / 300;
      ctx.strokeRect(x, y, right - x, bottom - y);
    }
  }
  ctx.globalAlpha = 1;
  $('count').textContent = instances.length;
  $('reset').hidden = state.selected === null;
  $('list').replaceChildren();
  if (!instances.length) {
    const tip = document.createElement('p');
    tip.className = 'no-match';
    tip.textContent = '当前分数下没有预测实例。';
    $('list').append(tip);
  }
  for (const inst of instances) {
    const button = document.createElement('button');
    button.className = 'item';
    button.setAttribute('aria-pressed', String(inst.query === state.selected));
    button.title = '单独查看这个实例，再点一次恢复全部';
    const dot = document.createElement('span');
    dot.className = 'dot';
    dot.style.background = colors[inst.category];
    const label = document.createElement('span'),
      name = document.createElement('strong'),
      id = document.createElement('small');
    name.textContent = state.categories[inst.category];
    id.textContent = 'Query ' + String(inst.query).padStart(3, '0');
    label.append(name, id);
    const score = document.createElement('span');
    score.className = 'score';
    score.textContent = inst.score.toFixed(3);
    button.append(dot, label, score);
    button.onclick = () => {
      state.selected = state.selected === inst.query ? null : inst.query;
      draw();
    };
    $('list').append(button);
  }
}
async function prepare(data) {
  if (data.model.release_id !== state.model.release_id || data.model.checkpoint !== state.model.checkpoint ||
      data.model.threshold !== state.model.threshold) {
    throw new Error('工作台版本已更新，请刷新页面。');
  }
  const [original, instances] = await Promise.all([
    loadImage(data.image),
    Promise.all(
      data.instances.map(async (inst) => {
        const mask = await loadImage(inst.mask);
        const layer = document.createElement('canvas');
        layer.width = mask.width;
        layer.height = mask.height;
        const ctx = layer.getContext('2d');
        ctx.drawImage(mask, 0, 0);
        ctx.globalCompositeOperation = 'source-in';
        ctx.fillStyle = colors[inst.category];
        ctx.fillRect(0, 0, layer.width, layer.height);
        return { ...inst, layer };
      })
    )
  ]);
  return { size: data.size, original, instances };
}
function show(result) {
  state.result = result;
  state.selected = null;
  $('photo').src = result.original.src;
  $('empty').hidden = true;
  $('visual').hidden = false;
  $('dimensions').textContent = result.size.join(' × ') + ' px';
  draw();
}
function film() {
  const start = Math.floor(state.index / 8) * 8;
  $('film').replaceChildren();
  for (let i = start; i < Math.min(start + 8, state.ids.length); i++) {
    const thumb = document.createElement('button');
    thumb.className = 'thumb';
    thumb.setAttribute('aria-label', '查看第 ' + (i + 1) + ' 张');
    thumb.setAttribute('aria-current', String(i === state.index));
    const im = document.createElement('img');
    im.alt = '';
    im.src = '/photo/' + state.ids[i];
    thumb.append(im);
    thumb.onclick = () => selectImage(i);
    $('film').append(thumb);
  }
}
async function selectImage(index) {
  if (!Number.isInteger(index) || !state.ids.length) return;
  state.index = Math.min(state.ids.length - 1, Math.max(0, index));
  const version = ++state.version;
  const iid = state.ids[state.index];
  clear('正在读取预测…');
  $('photoName').textContent = '图像 ' + iid;
  $('page').value = state.index + 1;
  $('prev').disabled = state.index === 0;
  $('next').disabled = state.index === state.ids.length - 1;
  film();
  try {
    const data = await jsonResponse(await fetch('/api/example/' + iid));
    const result = await prepare(data);
    if (version !== state.version) return;
    show(result);
    message('已保存的 ' + $('modelVersion').textContent + ' 预测');
  } catch (err) {
    if (version === state.version) clear(err.message);
  }
}
function uploadView() {
  $('photoName').textContent = state.file ? state.file.name : '选择一张服饰照片';
  $('run').disabled = !state.file || state.busy;
  $('file').disabled = state.busy;
  if (state.uploaded) {
    show(state.uploaded);
    message($('modelVersion').textContent + ' · CPU 推理完成');
    return;
  }
  clear(state.file ? '准备分割' : '选择图片后可在这里对照原图与预测。');
  if (state.file) {
    $('photo').src = state.preview;
    $('visual').hidden = false;
    $('empty').hidden = true;
    const cv = $('canvas');
    cv.width = 600;
    cv.height = 300;
    const ctx = cv.getContext('2d');
    ctx.fillStyle = '#55645c';
    ctx.font = '18px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(state.busy ? '正在分割…' : '点击「开始分割」查看结果', 300, 155);
    message(state.busy ? '项目服务器 CPU 正在处理…' : '待提交');
  }
}
function mode(value) {
  state.mode = value;
  ++state.version;
  const gallery = value === 'gallery';
  $('galleryMode').setAttribute('aria-pressed', String(gallery));
  $('uploadMode').setAttribute('aria-pressed', String(!gallery));
  $('source').textContent = gallery ? 'FASHIONPEDIA / DEV' : 'CUSTOM IMAGE';
  $('film').hidden = !gallery;
  $('paging').hidden = !gallery;
  $('uploadPanel').hidden = gallery;
  if (gallery) selectImage(state.index);
  else uploadView();
}
$('galleryMode').onclick = () => mode('gallery');
$('uploadMode').onclick = () => mode('upload');
$('next').onclick = () => selectImage(state.index + 1);
$('prev').onclick = () => selectImage(state.index - 1);
$('page').onchange = () => selectImage(Number($('page').value) - 1);
$('page').onkeydown = (e) => {
  if (e.key === 'Enter') $('page').blur();
};
document.addEventListener('keydown', (e) => {
  if (state.mode !== 'gallery' || ['INPUT', 'BUTTON'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowRight') selectImage(state.index + 1);
  if (e.key === 'ArrowLeft') selectImage(state.index - 1);
});
$('threshold').oninput = () => {
  $('thresholdValue').value = (Number($('threshold').value) / 100).toFixed(2);
  draw();
};
$('boxes').onchange = draw;
$('reset').onclick = () => {
  state.selected = null;
  draw();
};
$('file').onchange = () => {
  const file = $('file').files[0];
  if (!file) return;
  if (state.preview) URL.revokeObjectURL(state.preview);
  state.file = null;
  state.uploaded = null;
  state.preview = null;
  if (file.size > 20 * 1024 * 1024) {
    uploadView();
    message('请选择 20 MB 以内的图片。');
    return;
  }
  state.file = file;
  state.preview = URL.createObjectURL(file);
  uploadView();
};
$('run').onclick = async () => {
  if (!state.file || state.busy) return;
  state.uploaded = null;
  state.busy = true;
  uploadView();
  let error = '';
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'X-Viewer-Token': state.token },
      body: state.file
    });
    state.uploaded = await prepare(await jsonResponse(response));
  } catch (err) {
    error = err.message;
  } finally {
    state.busy = false;
  }
  if (state.mode === 'upload') {
    uploadView();
    if (error) message(error);
  }
};
async function start() {
  const data = await jsonResponse(await fetch('/api/gallery'));
  state.model = data.model;
  $('modelVersion').textContent = data.model.completed_updates.toLocaleString('en-US') + ' 步';
  const minimum = Math.round(data.model.threshold * 100);
  $('threshold').min = minimum;
  $('threshold').value = minimum;
  $('threshold').disabled = false;
  $('thresholdValue').value = data.model.threshold.toFixed(2);
  $('minimumScore').textContent = data.model.threshold.toFixed(2);
  state.ids = data.ids;
  state.token = data.token;
  for (const c of data.categories) state.categories[c.id] = c.name_zh;
  $('total').textContent = data.ids.length;
  $('page').max = data.ids.length;
  mode('gallery');
}
start().catch((err) => clear(err.message));
