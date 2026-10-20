/* 对输出 SVG 的绘制属性分组；保留路径顺序、透明度、渐变与滤镜。 */
const SVGColorLayers = (() => {
  function hex(value) {
    const color = (value || '').trim().toLowerCase();
    if (color === 'black') return '#000000';
    if (color === 'white') return '#FFFFFF';
    if (/^#[\da-f]{6}$/.test(color)) return color.toUpperCase();
    if (/^#[\da-f]{3}$/.test(color)) return `#${[...color.slice(1)].map(c => c + c).join('')}`.toUpperCase();
    const probe = document.createElement('span');
    probe.style.color = color;
    const rgb = probe.style.color.match(/^rgb\((\d+),\s*(\d+),\s*(\d+)\)$/);
    return rgb ? `#${rgb.slice(1).map(v => Number(v).toString(16).padStart(2, '0')).join('')}`.toUpperCase() : null;
  }

  function inherited(node, property, fallback) {
    for (let current = node; current; current = current.parentElement) {
      const value = current.style?.getPropertyValue(property) || current.getAttribute(property);
      if (value && value !== 'inherit') return value.trim();
    }
    return fallback;
  }

  function parse(svg) {
    const doc = new DOMParser().parseFromString(svg, 'image/svg+xml');
    if (doc.querySelector('parsererror') || doc.documentElement.localName !== 'svg') return null;
    const groups = new Map();
    const originals = new Map();
    function add(node, property, paint, kind) {
      if (!paint || paint === 'none' || paint === 'transparent') return;
      const color = hex(paint);
      const gradientRef = paint.match(/^url\(\s*['"]?#([^)'"\s]+)['"]?\s*\)$/);
      const key = `${kind}:${color || paint}`;
      if (!groups.has(key)) {
        let swatch = color || '#A8B4C9';
        if (gradientRef) {
          const gradient = [...doc.querySelectorAll('[id]')].find(n => n.id === gradientRef[1]);
          const stops = [...(gradient?.querySelectorAll('stop') || [])]
            .map(stop => hex(stop.style?.getPropertyValue('stop-color') || stop.getAttribute('stop-color'))).filter(Boolean);
          if (stops.length) swatch = `linear-gradient(135deg,${stops.join(',')})`;
        }
        groups.set(key, { id: groups.size, color, originalColor: color, paint, kind,
          gradient: !!gradientRef, swatch, visible: true, targets: [] });
      }
      groups.get(key).targets.push({ node, property });
      if (!originals.has(node)) originals.set(node, [...node.attributes].map(a => [a.name, a.value]));
    }
    for (const node of doc.querySelectorAll('path,rect,circle,ellipse,polygon,polyline,line,text,use')) {
      if (node.closest('defs,clipPath,mask,pattern,symbol')) continue;
      const fill = inherited(node, 'fill', 'black');
      // SVG line 没有填充区域，避免制造不可见的默认黑色层。
      if (node.localName !== 'line') add(node, 'fill', fill, 'fill');
      add(node, 'stroke', inherited(node, 'stroke', 'none'), 'stroke');
    }
    for (const node of doc.querySelectorAll('feFlood,feDropShadow')) {
      add(node, 'flood-color', inherited(node, 'flood-color', 'black'), 'effect');
    }
    return { original: svg, doc, originals, layers: [...groups.values()] };
  }

  function changed(model) {
    return !!model?.layers.some(layer => !layer.visible || layer.color !== layer.originalColor);
  }

  function serialize(model) {
    if (!changed(model)) return model.original;
    // 每次从原始属性重建，隐藏/显示、改色/恢复不会累积误差或丢失原透明度。
    for (const [node, attributes] of model.originals) {
      for (const attr of [...node.attributes]) node.removeAttribute(attr.name);
      for (const [name, value] of attributes) node.setAttribute(name, value);
    }
    for (const layer of model.layers) {
      if (layer.visible && layer.color === layer.originalColor) continue;
      for (const { node, property } of layer.targets) {
        if (layer.color && layer.color !== layer.originalColor) {
          node.setAttribute(property, layer.color);
          node.style.setProperty(property, layer.color);
        }
        if (!layer.visible) {
          const name = layer.kind === 'effect' ? 'flood-opacity' : property;
          const value = layer.kind === 'effect' ? '0' : 'none';
          node.setAttribute(name, value);
          node.style.setProperty(name, value);
        }
      }
    }
    return new XMLSerializer().serializeToString(model.doc);
  }

  return { parse, serialize, changed };
})();
