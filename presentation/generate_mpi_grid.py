N=6

head ="""
```{=html}
<div style="display:flex; flex-direction:row; justify-content: space-around; margin:50px;">
        """
bot="""
</div>
```"""

names   = ["tl","tt","tr","ll","dt","rr","bl","bb","br"]
name_op = ["br","bb","bl","rr","dt","ll","tr","tt","tl"]
corner  = [names[i][0] != names[i][1] for i in range(9)]

def add_grid(init, pref, exch=False, d=0, inner=False, draw_arrows=True):
    res = init + """<div style="display:flex; flex-direction:column; align-items:center;">
    <div style="
    width: 400px;
    aspect-ratio:1;
    margin: 0;
    margin-left:auto;
    margin-right:auto;
    grid-template-columns: """+f"{"1fr " * N}"+""";
    display: grid;
    position:relative;
    background: #eee;" data-id="pushbox">"""
    for y in range(N):
        for x in range(N):
            def set(d):
                outer = x==0 or x==N-1 or y==0 or y==N-1
                l = x == d
                r = x == N-1-d
                t = y == d
                b = y == N-1-d
                tl = t and l
                bl = b and l
                tr = t and r
                br = b and r
                return outer,l,r,t,b,tl,bl,tr,br
            outer, l,r,t,b,tl,bl,tr,br = set(d)
            inner_active = x>=2 and x<=N-3 and y>=2 and y<=N-3
            inner_active_2 = x>=1 and x<=N-2 and y>=1 and y<=N-2
            # decide which arrows are active
            active = [t or l, t, t or r, l, False, r, b or l, b, b or r]
            if d==0:
                if tl:
                    active = [n=="tl" for n in names]
                if tr:
                    active = [n=="tr" for n in names]
                if bl:
                    active = [n=="bl" for n in names]
                if br:
                    active = [n=="br" for n in names]
            if outer and d>0:
                active = [False]*9
            dx = -(1 if (r or br or tr) else (
                -1 if (l or bl or tl) else 
                0
            ))
            dy = -(1 if (t or tr or tl) else (
                -1 if (b or br or bl) else 
                0
            ))
            color = (
                "#ddd" if outer and d>0 else (
                "#8fadcc" if (t or b) else (
                "#bd8fcc" if (l or r) else (
                "#ddd")))
            )
            # write result
            res += f"<div class='bb' {"style='background:#558855;'"if inner_active and inner else (
                "style='background:#885555;'" if inner_active_2 and inner else ""
                )}>"
            if draw_arrows:
                for i in range(9):
                    name = name_op[i] if exch else names[i]
                    color_inner = "#cc967a" if (corner[i] and active[i] and (
                        (tl and names[i]=="tl") or 
                        (tr and names[i]=="tr") or 
                        (bl and names[i]=="bl") or 
                        (br and names[i]=="br") 
                    )) else ("#bd8fcc" if 
                            ((active[i] and r and "r" in names[i]) or 
                            (active[i] and l and "l" in names[i]))
                            else color)
                    idh = f"asdf{x}{y}{i}"
                    if active[i]:
                        if "l" in name or "r" in name:
                            idh = f"{pref}{name}{y}"

                        if "t" in name and t and not exch:
                            idh = f"t{pref}{name}{x}"
                        if "t" in name and b and exch:
                            idh = f"t{not pref}{name}{x}"
                        
                        if "b" in name and b and not exch:
                            idh = f"b{pref}{name}{x}"
                        if "b" in name and t and exch:
                            idh = f"b{not pref}{name}{x}"
                        # if "r" in name or "l" in name:
                        #     idh = f"{pref}{name}{y}"
                        # elif ("t" in name or "b" in name) and (t or b):
                        #     idh = f"{not pref if exch else pref}{name}{x}"
                        # else:
                        #     idh = "?"
                    res += f"<img src='res/{name}.png' style='background: {color_inner};' class='arrow a{name} {"" if active[i] else "ina"}' {
                        # ""
                        (f"data-id='{idh}'") 
                        if active[i] else ""
                    }/>"
            res+="</div>"
    res += """
    </div>
</div>"""
    return res


# write first file
res = head
res = add_grid(res, False, d=1)
res = add_grid(res, True, d=1)
res += bot

with open("mpi-comm.qmd", "w") as f:
    f.write(res)

# write second file
res = head
res = add_grid(res, True, exch=True)
res = add_grid(res, False, exch=True)
res += bot

with open("mpi-comm2.qmd", "w") as f:
    f.write(res)

# write third file
N = 15
res = head
res = add_grid(res, True, exch=True, inner=True)
res += bot
with open("mpi-comm3.qmd", "w") as f:
    f.write(res)

# write fourth file
N = 50
res = head
res = add_grid(res, True, exch=True, inner=True, draw_arrows=False)
res += bot
with open("mpi-comm4.qmd", "w") as f:
    f.write(res)