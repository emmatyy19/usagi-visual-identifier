const shelf = document.querySelector('#shelf');
const search = document.querySelector('#search');
const sort = document.querySelector('#sort');
const count = document.querySelector('#result-count');
const empty = document.querySelector('#empty');
const clear = document.querySelector('#clear');
const dialog = document.querySelector('#doll-dialog');
let items = [];

function showDetails(item) {
  document.querySelector('#doll-name').textContent = item.name;
  document.querySelector('#doll-description').textContent = item.details;
  const image = document.querySelector('#doll-photo');
  image.src = item.photo;
  image.alt = item.name + ' Usagi doll from my collection';
  const cues = document.querySelector('#doll-cues');
  cues.replaceChildren(...item.cues.map(cue => {
    const element = document.createElement('li');
    element.textContent = cue;
    return element;
  }));
  dialog.showModal();
}

function card(item) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'doll-card';
  button.setAttribute('aria-label', 'Meet ' + item.name);
  const photo = document.createElement('div');
  photo.className = 'card-photo';
  const image = document.createElement('img');
  image.src = item.photo;
  image.alt = item.name + ' Usagi doll';
  image.width = 675;
  image.height = 900;
  image.loading = 'lazy';
  photo.append(image);
  const content = document.createElement('div');
  content.className = 'card-copy';
  const title = document.createElement('h3');
  title.textContent = item.name;
  const details = document.createElement('p');
  details.textContent = item.cues.join(' · ');
  const link = document.createElement('span');
  link.className = 'card-link';
  link.textContent = 'A closer look';
  const arrow = document.createElement('span');
  arrow.textContent = '↗';
  arrow.setAttribute('aria-hidden', 'true');
  link.append(arrow);
  content.append(title, details, link);
  button.append(photo, content);
  button.addEventListener('click', () => showDetails(item));
  return button;
}

function render() {
  const query = search.value.trim().toLocaleLowerCase();
  const visible = items.filter(item => [item.name, item.details, ...item.cues].join(' ').toLocaleLowerCase().includes(query));
  if (sort.value === 'name') visible.sort((a, b) => a.name.localeCompare(b.name));
  shelf.replaceChildren(...visible.map(card));
  count.textContent = `${visible.length} of ${items.length} dolls${query ? ' match your search' : ' on the shelf'}`;
  empty.hidden = visible.length !== 0;
  clear.hidden = !query;
}

function resetSearch() {
  search.value = '';
  render();
  search.focus();
}

search.addEventListener('input', render);
sort.addEventListener('change', render);
clear.addEventListener('click', resetSearch);
document.querySelector('#show-all').addEventListener('click', resetSearch);
document.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => {
  if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }
});

try {
  const response = await fetch('./collection.json', { cache: 'no-cache' });
  if (!response.ok) throw new Error('Collection unavailable');
  const collection = await response.json();
  items = collection.items;
  render();
} catch {
  document.querySelector('#load-error').hidden = false;
  search.disabled = true;
  sort.disabled = true;
}
