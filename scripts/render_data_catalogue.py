"""Render readable/CSV data handoff from the versioned JSON registry."""
import csv
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
registry=json.loads((root/'snnbench/resources/datasets.json').read_text());rows=registry['datasets']
fields=list(dict.fromkeys(k for row in rows for k in row))
with (root/'snnbench/resources/datasets.csv').open('w',newline='',encoding='utf-8-sig') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
    for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})
text=['# 全部数据来源与派生协议清单（'+str(len(rows))+'项）','',registry['count_meaning'],'',
      '这是来源/任务/素材目录，不能计作同数量的真实动态数据集或已完成实验。真实数据优先级、协议差异、作者预处理与分组限制见 [真实数据与模型扩展](REAL_DATA_AND_MODEL_EXPANSION.md)。','',
      '| ID | 名称 | 优先级 | 当前适配 | 官方/作者入口 |','|---|---|---|---|---|']
for row in rows:
    text.append('| '+' | '.join([row['id'],row['name'],row['priority'],row['adapter'],'[来源]('+row['source_url']+')'])+' |')
text+=['','## 名称相近但不可混同的原始版本','']
for row in rows:
    for source in row.get('related_raw_sources',[]):
        text.append('- '+row['id']+'关联原始版本：['+source['name']+']('+source['source_url']+')。参与者'+str(source.get('participants','未核对'))+'，类别'+str(source.get('classes','未核对'))+'；不自动等同该条目历史派生版本。')
text+=['','## 注册下载文件','',
       '默认跳过标成“作者原生、可选”的预处理包；显式使用 `--include-native` 获取。它们可能丢失受试者ID，不能自动用于本文分组协议。无供应方校验的文件只记录本地SHA256，不声称获得了供应方校验。','',
       '| 数据ID | 文件 | 默认/可选 | 链接 | 供应方校验 |','|---|---|---|---|---|']
for row in rows:
    for file in row.get('files',[]):
        text.append('| '+' | '.join([row['id'],file['filename'],'作者原生、可选' if file.get('optional') else '默认','[下载]('+file['url']+')',file.get('sha256',file.get('md5','未公布；下载时记录本地SHA256'))])+' |')
(root/'docs/DATASETS_ALL.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
print(json.dumps({'entries':len(rows),'registered_files':sum(len(r.get('files',[])) for r in rows)}))
