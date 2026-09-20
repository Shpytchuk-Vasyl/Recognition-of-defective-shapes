export class Tabs {
  constructor({ root, onChange }) {
    this.buttons = [...root.querySelectorAll('.tab')];
    this.panes = [...root.querySelectorAll('.tabpane')];
    this.onChange = onChange;
    this.current = this.buttons[0]?.dataset.tab;
    this.buttons.forEach(b => b.addEventListener('click', () => this.select(b.dataset.tab)));
  }

  select(name) {
    this.current = name;
    this.buttons.forEach(b => b.classList.toggle('is-active', b.dataset.tab === name));
    this.panes.forEach(p => p.classList.toggle('is-active', p.dataset.pane === name));
    this.onChange?.(name);
  }
}
