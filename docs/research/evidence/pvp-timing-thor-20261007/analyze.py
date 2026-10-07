import pathlib,csv,json,statistics,math
root=pathlib.Path("/var/tmp/remoteops-pvp-timing.nGx9ZQGx/campaign.mFjggf9M")
rows=[]; completes=[]; checks=0; wrong=0
for log in sorted(root.glob("block-1/run-*/timing.log")):
 for line in log.read_text().splitlines():
  if line.startswith("PVP_TIMING_COMPLETE,"): completes.append({"run":log.parent.name,"row":line})
  if line.startswith("PVP_TIMING_CHECK,"):
   checks+=1
   wrong+=int(line.split(",wrong=")[1].split(",")[0])
  if not line.startswith("PVP_TIMING_SAMPLE,"):continue
  r=dict(x.split("=",1) for x in line.split(",")[1:])
  if r["measured"]!="true":continue
  r["run"]=log.parent.name
  for k in ["K","G","trial","repeat","position","round","wall_ns","upload_ns"]:r[k]=int(r[k])
  rows.append(r)
groups={}
for r in rows:groups.setdefault((r["run"],r["K"],r["G"],r["bank"],r["phase"],r["arm"]),[]).append(r["wall_ns"])
summary=[]
for key,values in sorted(groups.items()):
 v=sorted(values)
 summary.append(dict(zip(["run","K","G","bank","phase","arm"],key),n=len(v),median_ns=statistics.median(v),p95_ns=v[math.ceil(.95*len(v))-1],min_ns=min(v),max_ns=max(v)))
if rows:
 with (root/"samples.csv").open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
if summary:
 with (root/"summary.csv").open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
data={"measured_samples":len(rows),"checks":checks,"wrong_words":wrong,"complete":completes,"exact_samples":sum(r["exact"]=="true" for r in rows),"admission":"none"}
(root/"analysis.json").write_text(json.dumps(data,indent=2)+"\n")
print(json.dumps(data))
for phase in ["forward","inverse"]:
 for arm in sorted(set(r["arm"] for r in rows)):
  v=[r["wall_ns"]/1e6 for r in rows if r["arm"]==arm and r["phase"]==phase]
  if v:print(phase,arm,"n",len(v),"pooled_median_ms",round(statistics.median(v),6))

lookup={(r["run"],r["K"],r["G"],r["bank"],r["phase"],r["repeat"],r["arm"]):r["wall_ns"] for r in rows}
ratios=[]
for r in rows:
 if r["arm"]=="vec4":continue
 key=(r["run"],r["K"],r["G"],r["bank"],r["phase"],r["repeat"],"vec4")
 if key in lookup:ratios.append({"run":r["run"],"K":r["K"],"G":r["G"],"bank":r["bank"],"phase":r["phase"],"arm":r["arm"],"repeat":r["repeat"],"baseline_ns":lookup[key],"candidate_ns":r["wall_ns"],"ratio":lookup[key]/r["wall_ns"]})
if ratios:
 with (root/"paired-ratios.csv").open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(ratios[0]));w.writeheader();w.writerows(ratios)
ratio_summary=[]
for k,g in sorted(set((r["K"],r["G"]) for r in ratios)):
 for phase in ["forward","inverse"]:
  for arm in sorted(set(r["arm"] for r in ratios)):
   group=[r for r in ratios if r["K"]==k and r["G"]==g and r["phase"]==phase and r["arm"]==arm]
   if not group:continue
   by_case={}
   for r in group:by_case.setdefault((r["run"],r["bank"]),[]).append(r["ratio"])
   medians=[statistics.median(v) for v in by_case.values()]
   ratio_summary.append({"K":k,"G":g,"phase":phase,"arm":arm,"n":len(group),"median_paired_ratio":statistics.median(r["ratio"] for r in group),"case_median_min":min(medians),"case_median_max":max(medians),"median_baseline_ms":statistics.median(r["baseline_ns"] for r in group)/1e6,"median_candidate_ms":statistics.median(r["candidate_ns"] for r in group)/1e6})
if ratio_summary:
 with (root/"ratio-summary.csv").open("w") as f:
  w=csv.DictWriter(f,fieldnames=list(ratio_summary[0]));w.writeheader();w.writerows(ratio_summary)
print("FORWARD_GEOMETRY_RATIOS")
for r in ratio_summary:
 if r["phase"]=="forward": print(json.dumps(r))
