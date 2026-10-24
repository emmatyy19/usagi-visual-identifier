// Keep native navigation and form validation, with immediate progress feedback.
(() => {
  const status = document.querySelector('.loading-status');
  const message = document.querySelector('.loading-message');
  const progress = document.querySelector('.page-progress');
  let pendingForm = null;
  let pendingButton = null;
  let originalLabel = '';
  let slowTimer;

  function reset() {
    clearTimeout(slowTimer);
    status.hidden = true;
    progress.hidden = true;
    if (pendingForm) pendingForm.removeAttribute('aria-busy');
    if (pendingButton) {
      pendingButton.removeAttribute('aria-disabled');
      pendingButton.textContent = originalLabel;
    }
    pendingForm = null;
    pendingButton = null;
  }

  function start(label, detail) {
    clearTimeout(slowTimer);
    message.textContent = label;
    status.hidden = false;
    progress.hidden = false;
    slowTimer = setTimeout(() => {
      message.textContent = detail || 'Still loading. You can use your browser’s reload button to try again if this takes too long.';
    }, 10000);
  }

  document.addEventListener('submit', (event) => {
    if (event.defaultPrevented) return;
    const form = event.target;
    if (!(form instanceof HTMLFormElement) || form.target || form.method === 'dialog') return;
    if (pendingForm === form) {
      event.preventDefault();
      return;
    }
    reset();
    pendingForm = form;
    form.setAttribute('aria-busy', 'true');
    const button = event.submitter;
    if (button instanceof HTMLButtonElement) {
      pendingButton = button;
      originalLabel = button.textContent;
      // aria-disabled preserves the submitter's name/value in native form data.
      button.setAttribute('aria-disabled', 'true');
      button.textContent = form.dataset.loadingButton || (form.method === 'post' ? 'Saving…' : 'Loading…');
    }
    start(form.dataset.loadingMessage || (form.method === 'post' ? 'Saving your changes…' : 'Loading results…'), form.dataset.loadingDetail);
  });

  document.addEventListener('click', (event) => {
    if (event.defaultPrevented || event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
    const link = event.target.closest('a[href]');
    if (!link || link.target || link.hasAttribute('download')) return;
    const url = new URL(link.href, location.href);
    if (url.origin !== location.origin || !['http:', 'https:'].includes(url.protocol)) return;
    if (url.pathname === location.pathname && url.search === location.search && url.hash) return;
    start('Opening page…');
  });

  // Restore controls when Back returns a page from the browser's memory cache.
  window.addEventListener('pageshow', reset);
  window.addEventListener('load', reset);
  if (document.readyState !== 'complete') start('Loading page…');
})();
