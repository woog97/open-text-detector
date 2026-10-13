type Box = { id: number; x: number; y: number; w: number; h: number };

/** Called for one rendered page, in screen pixels. Never joins passage IDs. */
export function mergeHighlightBoxes<T extends Box>(boxes: T[]): T[] {
  const passages = new Map<number, T[]>();
  for (const box of boxes) {
    if(![box.x,box.y,box.w,box.h].every(Number.isFinite)||box.w<=0||box.h<=0)continue;
    const group = passages.get(box.id) || [];
    group.push(box);
    passages.set(box.id, group);
  }
  const merged: T[] = [];
  for (const group of passages.values()) {
    const lines: { anchor: T; boxes: T[] }[] = [];
    let active: typeof lines = [];
    for (const box of [...group].sort((a, b) => a.y - b.y || a.x - b.x)) {
      // Compare to the original line anchor so small offsets cannot accumulate
      // into a join across separate lines.
      active=active.filter(({anchor})=>anchor.y+anchor.h>=box.y);
      const line = active.find(({ anchor }) =>
        Math.abs(box.y + box.h / 2 - anchor.y - anchor.h / 2) <= Math.min(box.h, anchor.h) * .3,
      );
      if (line) line.boxes.push(box);
      else {const next={anchor:box,boxes:[box]};lines.push(next);active.push(next);}
    }
    for (const line of lines) {
      let current: T | undefined;
      for (const box of line.boxes.sort((a, b) => a.x - b.x)) {
        // Fill ordinary word spaces, but leave column gutters and large gaps.
        if (current && box.x - (current.x + current.w) <= Math.min(line.anchor.h, box.h) * .65) {
          const right = Math.max(current.x + current.w, box.x + box.w);
          const bottom = Math.max(current.y + current.h, box.y + box.h);
          current.y = Math.min(current.y, box.y);
          current.w = right - current.x;
          current.h = bottom - current.y;
        } else {
          current = { ...box };
          merged.push(current);
        }
      }
    }
  }
  return merged;
}
