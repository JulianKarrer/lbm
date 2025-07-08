import colorsys

N=2
S=25

def hsv(h,s,v):
    # https://stackoverflow.com/questions/78819924/how-to-convert-hsv-values-to-hexcode-format
    r, g, b = (int(255 * component) for component in colorsys.hsv_to_rgb(h / 360, s / 100, v / 100))
    return f"#{r:02x}{g:02x}{b:02x}"

head ="""
```{=html}
<div style="display:flex; flex-direction:row; justify-content: space-around; margin:50px;">
        """
bot="""
</div>
```"""

names = ["tl","tt","tr","ll","dt","rr","bl","bb","br"]
cols  = [hsv(360/9*i,20,90) for i in range(9)]

def add_grid(init, onGrid=True, hints=None, soa=False):
    res = init + """<div style="display:flex; flex-direction:column; align-items:center;">
    <div style="
    width: 300px;
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
            res += f"<div class='bb'>"
            if onGrid:
                for i in range(9):
                    name = names[i]
                    res += f"<img src='res/{name}.png' style='background:{cols[i]}' class='arrow a{name}' data-id='{x}{y}{i}'/>"
            res+="</div>"
    res += """
    </div>
    <div style="display:flex; flex-direction:row; align-items:center; margin-top:50px; position:relative;">
        """
    if soa:
        for i in range(9):
            for y in range(N):
                for x in range(N):
                    name = names[i]
                    res+= f"""<div class='bb' style='width:{S}px; height:{S}px; background:white;'>{
                        f"<img src='res/{name}.png' style='background:{cols[i]}'  class='full-arrow' data-id='{x}{y}{i}'/>" 
                        if not onGrid else ""
                    }</div>"""
    else:
        for y in range(N):
            for x in range(N):
                for i in range(9):
                    name = names[i]
                    res+= f"""<div class='bb' style='width:{S}px; height:{S}px; background:white;'>{
                        f"<img src='res/{name}.png' style='background:{cols[i]}'  class='full-arrow' data-id='{x}{y}{i}'/>" 
                        if not onGrid else ""
                    }</div>"""
    if not hints is None:
        for j in range(N*N):
            if soa:
                res += f"<img src='res/tt.png' style='filter:invert();width:{S}px;height:{S}px;position:absolute;top:{S}px;left:{(j+hints*N*N)*S*1.08}px' data-id='hint{j}'/>" 
            else:
                res += f"<img src='res/tt.png' style='filter:invert();width:{S}px;height:{S}px;position:absolute;top:{S}px;left:{S*1.08*(hints+j*9)}px' data-id='hint{j}'/>" 
    res+="""
    </div>
</div>"""
    return res


# AOS
with open("aos-packing.qmd", "w") as f:
    f.write(add_grid(head) + bot)

with open("aos-packing2.qmd", "w") as f:
    f.write(add_grid(head, False, hints=0) + bot)

with open("aos-packing3.qmd", "w") as f:
    f.write(add_grid(head, False, hints=1) + bot)

with open("aos-packing4.qmd", "w") as f:
    f.write(add_grid(head, False, hints=2) + bot)

with open("aos-packing5.qmd", "w") as f:
    f.write(add_grid(head, False, hints=3) + bot)

# SOA
with open("soa-packing.qmd", "w") as f:
    f.write(add_grid(head, soa=True) + bot)

with open("soa-packing2.qmd", "w") as f:
    f.write(add_grid(head, False, hints=0, soa=True) + bot)

with open("soa-packing3.qmd", "w") as f:
    f.write(add_grid(head, False, hints=1, soa=True) + bot)

with open("soa-packing4.qmd", "w") as f:
    f.write(add_grid(head, False, hints=2, soa=True) + bot)

with open("soa-packing5.qmd", "w") as f:
    f.write(add_grid(head, False, hints=3, soa=True) + bot)