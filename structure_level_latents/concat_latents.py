import json

f1 = "/home/mehuldarak/athena/structure_level_latents/structure_level_latents_309machine.json"
f2 = "/home/mehuldarak/athena/structure_level_latents/structure_level_latents_polaris.json"
out = "/home/mehuldarak/athena/structure_level_latents/structure_level_latents_all.json"

with open(f1) as a:
    d1 = json.load(a)

with open(f2) as b:
    d2 = json.load(b)

assert isinstance(d1, list) and isinstance(d2, list)

d_all = d1 + d2

with open(out, "w") as f:
    json.dump(d_all, f, indent=2)

print(f"Written {len(d_all)} entries to {out}")