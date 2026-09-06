"""Render the computed FSM map as reproducible PNG/SVG and a Korean report."""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Rectangle, Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_fsm_v4_feasibility import OUT, cfg, HALF

COLORS = ['#463451', '#e87977', '#f5bc58', '#57b7b0', '#248666', '#b4bdc9']
NAMES = ['회전만으로 불가¹', '현 로직 실패²', '가능 · 어려움', '가능 · 보통', '가능 · 쉬움', '초기 관측 미평가']


def configure():
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False,
                         'font.size': 11, 'axes.edgecolor': '#ccd2d9',
                         'axes.labelcolor': '#374151', 'text.color': '#172233',
                         'xtick.color': '#596477', 'ytick.color': '#596477',
                         'savefig.facecolor': '#f7f9fc', 'figure.facecolor': '#f7f9fc'})


def edges(values, lower, upper):
    return np.r_[lower, (values[:-1]+values[1:])/2, upper]


def draw(ax, payload, complete=True, causes=False):
    radii, angles = np.array(payload['radii']), np.array(payload['angles'])
    values = np.array([r['category'] for r in payload['results']]).reshape(len(radii), len(angles))
    rr, aa = np.meshgrid(edges(radii, .75, 10.), np.radians(edges(angles, -45., 45.)), indexing='ij')
    cmap = ListedColormap(COLORS)
    norm = BoundaryNorm(np.arange(-.5, 6.5), 6)
    for rotation in ([0, 90, 180, 270] if complete else [0]):
        ang = aa + math.radians(rotation)
        ax.pcolormesh(rr*np.sin(ang), rr*np.cos(ang), values, cmap=cmap, norm=norm,
                      shading='flat', rasterized=True, zorder=1)
    ax.add_patch(Circle((0, 0), .75, color=COLORS[0], zorder=2))
    ax.add_patch(Rectangle((-HALF, -HALF), 2*HALF, 2*HALF, facecolor='#253144',
                           edgecolor='white', linewidth=1.5, zorder=8))
    for r in [2, 4, 6, 8, 10]:
        ax.add_patch(Circle((0, 0), r, fill=False, color='#ffffff', alpha=.55,
                           lw=.8, zorder=4))
    for a in ([-135, -45, 45, 135] if complete else [-45, 45]):
        b = math.radians(a)
        ax.plot([0, 10*math.sin(b)], [0, 10*math.cos(b)], color='white',
                lw=1., alpha=.8, ls='--', zorder=5)
    ax.set_aspect('equal')
    ax.set_facecolor('#edf1f6')
    ax.grid(alpha=.15, zorder=3)
    ax.set_xlabel('X · 팔레트 중심 기준 좌우 위치 (m)', labelpad=8)
    ax.set_ylabel('Y · 팔레트 중심 기준 위치 (m)', labelpad=8)
    if complete:
        ax.set(xlim=(-10.4,10.4), ylim=(-10.4,10.4), xticks=np.arange(-10,11,2), yticks=np.arange(-10,11,2))
        for a in [-30, 0, 30, 90, 180, 270]:
            b=math.radians(a)
            x,y=6.7*math.sin(b),6.7*math.cos(b)
            ax.annotate('', xy=(x-.5*math.sin(b), y-.5*math.cos(b)), xytext=(x,y),
                        arrowprops={'arrowstyle':'-|>', 'lw':1.4, 'color':'#23364a'}, zorder=10)
        ax.text(0, -.08, '팔레트', ha='center', va='center', color='white', fontsize=7, zorder=10)
    else:
        ax.set(xlim=(-3.5,3.5), ylim=(1.6,6.1), xticks=np.arange(-3,4), yticks=np.arange(2,7))
        b=np.linspace(-np.pi/4,np.pi/4,300)
        for z, style in [(2.2,'-'), (4.1,'--')]:
            r=z-cfg.CAMERA_TO_ROT_CENTER_Z_M+HALF*np.cos(b)
            ax.plot(r*np.sin(b), r*np.cos(b), style, color='#253144', lw=1.2, zorder=6)
        label_box={'facecolor':'white','alpha':.9,'edgecolor':'none','pad':3}
        ax.text(-.96,3.48,'카메라 Z = 2.2 m',fontsize=9,bbox=label_box,zorder=10)
        ax.text(-1.12,5.38,'카메라 Z = 4.1 m',fontsize=9,bbox=label_box,zorder=10)
        ax.text(-3.18,3.7,'-45°',fontsize=10,rotation=-45)
        ax.text(2.83,3.7,'+45°',fontsize=10,rotation=45)


def main():
    configure()
    data=json.loads((OUT/'results.json').read_text(encoding='utf-8'))
    rows=data['results']
    fig=plt.figure(figsize=(17,10), dpi=160)
    fig.text(.055,.943,'팔레트 주변 어디에서 삽입 계획이 성립하는가',fontsize=25,weight='bold')
    fig.text(.055,.902,'FSM v4  |  팔레트 중심을 바라보는 차량  |  회전 중심 거리 0–10 m  |  각 면 정면 기준 -45°…+45°',fontsize=12,color='#536174')
    ax=fig.add_axes([.055,.17,.445,.67])
    draw(ax,data)
    ax.set_title('4개 대칭 면 · 전체 배치',loc='left',fontsize=14,pad=12,weight='bold')
    zoom=fig.add_axes([.565,.435,.385,.4])
    draw(zoom,data,False)
    zoom.set_title('한 면 확대 · 근거리 전환 경계',loc='left',fontsize=14,pad=12,weight='bold')
    fig.text(.565,.353,'색상의 의미',fontsize=13,weight='bold')
    fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(COLORS,NAMES)],
               loc='upper left',bbox_to_anchor=(.557,.344),ncol=2,frameon=False,
               fontsize=11,labelspacing=1.05,handlelength=1.4,columnspacing=1.8)
    fig.text(.565,.17,'¹ 현재 yaw·포크 조건에서 제자리 회전만으로 해결 불가\n² 현재 계획·시야·시간 한도로 실패. 모든 경로의 불가능을 뜻하지 않음.\n성공 색상 = 이상적인 자세·동작에서 삽입 승인까지 도달',
             fontsize=10,color='#536174',linespacing=1.8)
    fig.text(.055,.075,f"계산: {len(rows):,}개 초기 조건 / 거리 0.1 m × 각도 2.5°  ·  4개 면은 90° 회전 복제  ·  차량 방향은 화살표처럼 중심을 향함",fontsize=10,color='#536174')
    fig.text(.055,.048,'전제: 안정된 PnP, 장애물 없음, 이상적 정지 위치. 수직 시야·차체 충돌·실제 인식률은 미평가. 색상은 성공확률이 아닙니다.',fontsize=10,color='#536174')
    fig.savefig(OUT/'feasibility_heatmap.png',dpi=160)
    fig.savefig(OUT/'feasibility_heatmap.svg')
    plt.close(fig)

    lookup={(round(r['radius_m'],3),round(r['beta_deg'],3)):r for r in rows}
    selected_r=[2.,2.5,3.,3.5,4.,5.,6.,8.,9.,10.]
    selected_a=[0.,10.,15.,20.,30.,45.]
    table=['| 중심 간 거리 | 0° | ±10° | ±15° | ±20° | ±30° | ±45° |',
           '|---:|---|---|---|---|---|---|']
    plain=['회전만으로 불가','FSM 실패','가능·어려움','가능·보통','가능·쉬움','관측 미평가']
    for r in selected_r:
        table.append('| '+f'{r:g} m'+' | '+' | '.join(plain[lookup[(r,a)]['category']] for a in selected_a)+' |')
    matched=sum(lookup[(r,a)]['category']==lookup[(r,-a)]['category']
                for r in data['radii'] for a in data['angles'] if a>0)
    pairs=sum(a>0 for a in data['angles'])*len(data['radii'])
    stats='\n'.join(f'- {plain[k]}: {sum(row["category"]==k for row in rows):,}점' for k in range(6))
    failures='\n'.join(f'- `{reason}`: {count}점' for reason,count in sorted(data['reason_counts'].items(),key=lambda it:-it[1]))
    possible=[r for r in rows if r['success']]
    boundaries=[]
    for r in [3.,4.,5.,6.,8.,9.]:
        successes=[row for row in rows if row['radius_m']==r and row['success']]
        max_angle=max((abs(row['beta_deg']) for row in successes),default=0.)
        boundaries.append(f'- 중심 거리 {r:g} m: 성공 표본의 최대 측면각 {max_angle:g}°. 그 사이의 모든 표본이 성공한다는 뜻은 아님.')
    report=f'''# FSM v4 팔레트 중심 좌표계 삽입 가능성 분석

![계산 히트맵](feasibility_heatmap.png)

현재 소스의 계획·삽입 판단을 재사용해 {len(rows):,}개 초기 조건을 계산했다. 이 결과는 **삽입 승인(READY_TO_INSERT)까지의 이상적 가능성**이며 실차 성공확률이나 모든 경로에 대한 도달가능성 증명은 아니다. CAN 송신 없이 계산했으며 운용 설정은 변경하지 않았다.

## 좌표와 대칭 가정

- 원점: 1.1 × 1.1 m 팔레트의 **몸체 중심**. 지도 점: **차량 회전 중심**.
- r: 두 중심 간 거리. β: 선택한 면의 정면 법선으로부터 차량 위치의 측면각(±45°).
- 차량은 팔레트 몸체 중심을 바라본다. yaw는 초기 β와 같다. 회전 후 yaw가 ±45° 안에 머문다는 추가 제한은 없다.
- 먼저 한 면의 좌우를 모두 계산한 뒤 90°씩 회전해 4면을 표시했다. 다른 면으로 갈아타는 경로는 탐색하지 않는다.
- 4면의 삽입구와 치수도 동일하다는 사용자 가정이다. 실제 PnP의 면 선택·90° 대칭 전환 오차는 반영하지 않았다.
- 초기 좌표 변환: 카메라 X = −0.55 sinβ, 카메라 Z = r − 0.55 cosβ − 0.68.
- 따라서 **정면 카메라 Z=2.2 m는 지도 r=3.43 m**, Z=4.1 m는 r=5.33 m다. 지도 거리 2.2 m와 코드 임계값 2.2 m는 다르다.

## 계산한 로직과 근사

`CalibrationFSMV4.step()`의 실제 정지 상태 분기, `_route_start_distance()`, `plan_waypoint()`, `safe_straight_continuation_m()`, `insertion_alignment_turn()`, 회전 후 검증 및 보정 예산을 사용했다. CAN 실행기는 생성하지 않았다.

실제 이동·센서 시간열을 생성하는 대신 계획한 회전과 직진의 이상적 종점을 연결하는 이벤트 시뮬레이션이다. 카메라와 회전 중심의 0.68 m 오프셋을 매 회전마다 적용한다. waypoint 직진 중 카메라 Z=2.2 m 또는 staging 평면에 먼저 도달하면 멈추고, 실제 FSM의 정지 후 판단을 수행한다. 초기 탐색은 제외하고 안정된 전체 앞면 관측에서 시작한다. 최초 전체 앞면이 안전 시야에 들어오지 않는 표본은 관측 미평가로 구분한다.

회전은 요구 각도가 정확히 달성된다고 가정한다. 시간은 현재 회전 응답 및 전진 적합식에 STOP 2초 + 10 Hz에서 안정 프레임 10개(1초)를 더한 참고값이다. 측정 오차, 관성, 실제 정지 위치, 재인식 시간, 디버깅 대기, 세션 회전 적응은 시뮬레이션하지 않았다. 직진 predictive STOP의 조기 정지/관성은 중간 waypoint에는 반영하지 않고 정확한 목표 도달로 근사했다. 따라서 실제 로그 재생 결과와 경계가 달라질 수 있다.

4 m 접근 동작에는 실제 15초 상한과 현재 가속·등속 적합을 적용했다. 15초 최대 적합 이동거리는 약 3.83 m이며 predictive STOP 0.12 m까지 낙관적으로 허용한다. 이 범위를 초과하는 원거리 표본은 시간 제한 실패로 표시한다. 동적 정지 동작을 재현한 것은 아니다.

55° 수평 FOV와 좌우 8% 여백을 사용한다. 실제 영상의 수직 시야, 카메라 높이·pitch, 장애물·차체·포크 전체 충돌, 진입구 깊이 방향 간섭, 10 m 검출률은 평가하지 않았다. 삽입 이후 DONE까지의 운동·시야 유지도 평가하지 않았다.

## 색상 판정

- **진보라 / 회전만으로 불가:** 근거리에서 현재 yaw ±20°와 포크 진입선 조건을 만족시킬 수 없다는 필요조건 위반. 후진·재배치나 다른 면 접근까지 불가능하다는 뜻은 아니다.
- **연빨강 / FSM 실패:** 현재 후보 탐색·시야·예산·시간 조건에서 삽입 승인을 못 받음. 물리적 절대 불가능과 구분한다.
- **초록 / 쉬움:** 누적 회전 ≤10°, waypoint ≤3회, 최종 한쪽 최소 포크 여유 ≥20 mm, 경로 최소 수평 시야 여유 ≥3°, 삽입 승인까지 참고 시간 ≤60초.
- **청록 / 보통:** 쉬움 외에 누적 회전 ≤35°, waypoint ≤5회, 포크 여유 ≥10 mm, 시야 여유 ≥1°, 참고 시간 ≤90초.
- **노랑 / 어려움:** 승인되지만 위 여유 조건을 만족하지 못함. 작은 오차에 민감하거나 보정 부담이 크다.
- **회색 / 초기 관측 미평가:** 전체 앞면이 처음부터 안전 시야에 들어온다는 가정 불충족. 실제 검색·시야 회복 성공 여부를 별도로 검증해야 한다.

쉬움/보통의 기준은 이 분석에서 명시적으로 정한 비교 기준이며 FSM 내부의 성공확률 모델이 아니다. 특히 플래너가 시야 한계까지 전진하거나 가장 작은 회전을 선택하므로, 계획 성공이어도 여유가 거의 0인 표본이 존재한다.

## 제자리 회전 불가 영역의 계산 근거

회전 중심의 선택한 면 기준 좌우 위치 u, 면까지 법선 거리 d, 최종 yaw α에 대해 현재 진입선 조건은 다음과 같다.

```text
|u − d tanα| + 0.30 / cosα ≤ 0.355
d ≥ 1.86 cosα + 0.30 |sinα|
|α| ≤ 20°
```

두 식은 각각 두 포크 바깥 진입선이 폭 0.71 m 구간 안에 들어오는 조건과 팔레트 면이 두 포크 끝보다 앞에 있는 조건이다. 여기서 d가 약 1.850 m보다 작거나, |u|가 d tan20° + 0.055 m보다 크면 어떤 허용 yaw도 조건을 만족할 수 없다. 이것은 충분한 불가능 판정이며 그 역은 주장하지 않는다. 근거리 진보라 영역만 이 증명을 적용했다. 도면의 r<0.75 m 영역도 이 필요조건으로 채웠다.

## 대표 계산 결과

{chr(10).join(table)}

{chr(10).join(boundaries)}

## 표본 분포와 실패 이유

{stats}

극좌표 균등 표본의 개수이며 지도 면적 비율·성공확률이 아니다. 거리 0.8–10 m를 0.1 m, 각도 −45–45°를 2.5° 간격으로 계산했다. 격자 사이를 추가 판정하거나 확률로 보간하지 않고 각 셀의 표본 색을 표시했다.

{failures}

## 검증과 재현

- 원본 시야 계산과 벡터화 계산을 300개 무작위 동작에 대조. 최대 차이 {data['vectorization_max_error']:.3g}°.
- 분석 전용 테스트 5개 통과: 좌표 역변환, 원본 시야식 대조, 제자리 회전 불가 필요조건, 원본 플래너와 가속 계산의 10개 대표 경로 일치, 거리별 대표 성공·실패. 불가능 판정은 해석식이며 각도 0.1° 순회로도 반례가 없는지 추가 확인했다.
- 좌우 대칭 표본의 등급 일치: {matched}/{pairs}쌍. 4면 대칭은 사용자 가정에 따라 시각적으로 회전 복제.
- `results.json`: 각 표본의 상태 경로·동작·최종 여유·실패 이유, 설정 스냅샷과 원본 파일 SHA-256.
- `samples.csv`: 표본별 좌표 및 계산 결과.
- `feasibility_heatmap.png` / `.svg`: 배포 가능한 이미지.

```powershell
python tools/visualize_fsm_v4_feasibility.py
python tools/render_fsm_v4_feasibility.py
python -m unittest discover -s tools -p test_fsm_v4_feasibility_analysis.py
```
'''
    (OUT/'README.md').write_text(report,encoding='utf-8')
    print('\n'.join(table))
    print('\n'.join(boundaries))
    print('symmetry',matched,pairs)
    print(OUT/'feasibility_heatmap.png')


if __name__ == '__main__':
    main()
