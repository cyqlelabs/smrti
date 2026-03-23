# Smrti-Town: Game Design Document

## 1. Opening Sequence

The game starts with an empty grass field. No buildings, no people, no UI clutter. Just terrain.

**Step 1 — Place the Town Hall.** The player taps/clicks the map to place the Town Hall. This is the only manual building placement in the entire game. The Town Hall appears as a real sprite on the isometric grid. This anchors the town's center.

**Step 2 — Choose the Mayor.** A selection screen presents 3 LLM-generated candidates (name, portrait description, short bio, personality traits). The player picks one. Gender, background, and governing style vary. The Mayor's personality directly biases all future council decisions (e.g., a fiscally conservative mayor will resist expensive proposals; a progressive one will prioritize education and culture early).

**Step 3 — Assemble the Council.** The Mayor appoints 4 advisors, each managing a domain:

| Role | Domain | Influences |
|------|--------|-----------|
| **Sheriff** | Security | Crime rate, patrol buildings, walls, jail |
| **Superintendent** | Education | Schools, libraries, universities, literacy rate |
| **Doctor** | Health | Clinics, hospitals, pharmacies, life expectancy |
| **Treasurer** | Finances | Tax rate, trade deals, budget allocation, debt |

These 5 people (Mayor + 4) are the founding council. They are the town's first residents. They live in the Town Hall initially (it has temporary quarters).

---

## 2. The Economic Foundation

### 2.1 Town Treasury

The town starts with a founding grant (e.g., 50,000 coins). Every action has a cost:

| Category | Examples | Cost Range |
|----------|----------|-----------|
| **Construction** | Buildings, roads, parks | 2,000-25,000 |
| **Salaries** | Council, teachers, doctors | 50-200/cycle |
| **Events** | Festivals, fairs, ceremonies | 500-5,000 |
| **Maintenance** | Building upkeep, road repair | 5% of build cost/year |
| **Emergency** | Fire, flood, epidemic response | Variable |

### 2.2 Revenue Streams

Money flows back in through:

- **Taxes** — Property tax (per house), business tax (per commercial building), income tax (% of citizen earnings). The Treasurer sets rates; too high = citizens leave, too low = treasury dries up.
- **Commerce** — Markets, shops, and businesses generate trade income. A percentage flows to the treasury.
- **Tourism** — Eventually, if the town has cultural buildings (museum, theater), visitors spend money.
- **Trade routes** — Once the town has a trading post, external trade generates passive income.

### 2.3 Citizen Economy

Each citizen has a personal wallet:

- **Income** — Earned from employment (job-specific wages), odd jobs if unemployed, or welfare if destitute.
- **Expenses** — Food, rent, goods, education fees, healthcare. These create demand that drives business creation.
- **Savings** — Citizens with surplus may eventually start businesses (entrepreneurship).

---

## 3. The Council Decision Loop

This is the core gameplay mechanic. Every N game-cycles, the council convenes.

### 3.1 The Proposal

The LLM generates a council meeting. Given the town's current state (population, buildings, treasury, citizen satisfaction, unmet needs), each advisor argues from their domain:

> **Sheriff:** "We have 12 residents and no constabulary. If we grow further without security, crime will rise."
> **Superintendent:** "The children have nowhere to learn. I propose a schoolhouse."
> **Treasurer:** "We have 38,000 coins. A school costs 8,000. I recommend we build housing first — more residents means more tax revenue."
> **Doctor:** "Two citizens fell ill last week. Without a clinic, we risk an outbreak."

The Mayor weighs these arguments (personality-biased) and proposes one action to the player. The player can:

1. **Approve** — Construction begins, money is deducted.
2. **Reject** — Council must reconvene with a different proposal.
3. **Counter-propose** — Player suggests an alternative from available options.

### 3.2 Available Buildings

The LLM chooses from a canonical building catalog included in its prompt. This ensures the backend always knows how to render and simulate the chosen building.

**Residential:**
- Cottage (2 residents, cheap)
- House (4 residents, moderate)
- Apartment (8 residents, expensive)
- Manor (2 residents, luxury, high tax)

**Commercial:**
- General Store (basic goods)
- Bakery (food production)
- Butcher (food production)
- Market (food + goods hub)
- Tavern (social + food)
- Inn (housing for newcomers + income)
- Blacksmith (tools, repairs)
- Tailor (clothing)
- Bookstore (education goods)

**Civic:**
- Town Hall (governance, already placed)
- School (basic education)
- Library (advanced education, research)
- University (professional training)
- Clinic (basic healthcare)
- Hospital (advanced healthcare)
- Church (spiritual needs, community)
- Courthouse (law, disputes)
- Fire Station (disaster prevention)
- Constabulary (security)
- Jail (crime reduction)

**Infrastructure:**
- Well / Water Tower (water supply)
- Granary (food storage, famine buffer)
- Warehouse (goods storage)
- Trading Post (external trade)
- Bridge (cross terrain)

**Cultural:**
- Park (recreation, free)
- Theater (entertainment, culture)
- Museum (culture, tourism)
- Festival Grounds (events)

**Industrial:**
- Lumber Mill (raw materials)
- Quarry (stone)
- Farm (food production, large)
- Windmill (grain processing)

Each building has: cost, maintenance, capacity, staff required, effects on nearby satisfaction, unlock conditions (population thresholds, prerequisite buildings, or petition triggers).

---

## 4. Population Growth

### 4.1 Immigration

People don't appear randomly. They're attracted by pull factors:

- **Housing availability** — No empty house = no new residents.
- **Employment** — Job openings pull working-age adults.
- **Services** — Families with children need schools; elderly need clinics.
- **Reputation** — A well-run town with low crime, good health, and culture attracts more.

When a house is built, the LLM generates the new residents: names, ages, family structure, skills, personality. A cottage might attract a young couple. An apartment might bring a family with children. A manor attracts a wealthy merchant.

### 4.2 Who Arrives

The type of building determines who comes:

| Building | Attracts |
|----------|---------|
| Cottage | Young couple, single worker |
| House | Small family (2 adults + 1-2 children) |
| Apartment | Mixed: singles, couples, small families |
| Manor | Wealthy individual/couple with specific skills |
| Inn | Temporary visitors who may settle if conditions are good |

### 4.3 Natural Growth

Existing couples may have children over time if:
- They're in the right age range
- They have stable housing
- Their satisfaction is above a threshold
- There's room in their home

---

## 5. Citizen Life Simulation

### 5.1 Needs Hierarchy

Every citizen has needs that drive behavior (inspired by Maslow):

| Priority | Need | Satisfied By |
|----------|------|-------------|
| 1 (urgent) | **Hunger** | Food sources (market, bakery, farm, home kitchen) |
| 2 | **Shelter** | Having a home |
| 3 | **Health** | Clinic/hospital when sick, general wellbeing |
| 4 | **Safety** | Low crime rate, constabulary presence |
| 5 | **Social** | Interacting with others, tavern, park, church |
| 6 | **Education** | School, library, university, bookstore |
| 7 | **Purpose** | Employment, meaningful work |
| 8 | **Culture** | Theater, museum, festivals, entertainment |
| 9 | **Self-actualization** | Starting a business, mastering a craft, leadership |

Lower needs dominate. A hungry citizen won't care about culture. An unsafe citizen won't pursue education.

### 5.2 Daily Routine

Citizens follow a daily cycle:
- **Morning** — Wake, eat (home or market/bakery), commute to work/school.
- **Daytime** — Work or study. Energy depletes. Skills improve if studying/practicing.
- **Evening** — Social time: visit tavern, park, church, or friends. Shop for goods.
- **Night** — Return home, sleep, restore energy.

Movement costs energy. Walking across town to reach a distant market is exhausting. This creates organic demand for distributed services — citizens will petition for a bakery on the east side if all food is on the west side.

### 5.3 Skill Development

Citizens develop skills over time through work and study:

| Skill Category | Learned At | Enables |
|---------------|-----------|---------|
| **Literacy** | School | Access to library, bookstore employment |
| **Medicine** | University + Clinic | Doctor role, better health outcomes |
| **Commerce** | Market work experience | Merchant role, business ownership |
| **Craftsmanship** | Workshop + apprenticeship | Blacksmith, tailor, builder roles |
| **Teaching** | University + Library | Teacher role, school quality |
| **Leadership** | Experience + high social | Council candidacy, business management |
| **Agriculture** | Farm work | Farmer efficiency, food yield |
| **Arts** | Theater + personal pursuit | Cultural contribution, tourism |

Skill progression is gradual. A child starts at school, gains literacy. A literate young adult can study at the university. A university graduate can practice medicine at the clinic. This creates a multi-generational pipeline.

### 5.4 Profession Activation

When a citizen gains enough skill, they can fill a professional role:

1. A building is built that requires staff (e.g., clinic needs a doctor).
2. The system checks if any citizen has the required skill level.
3. If yes, they're hired. If no, the building operates at reduced capacity until someone qualifies or a skilled immigrant arrives.
4. Citizens with high commerce + savings may petition to open their own business.

---

## 6. The Petition System

Petitions are the democratic pulse of the town. They come from two sources.

### 6.1 Council Petitions (Top-Down)

The council identifies systemic needs:
- "Population growing — we need more housing."
- "Crime is rising — build a constabulary."
- "Tax revenue declining — attract commerce."

### 6.2 Citizen Petitions (Bottom-Up)

Citizens generate petitions based on unmet needs:
- "I'm walking 20 minutes to buy bread. We need a bakery nearby." (movement exhaustion -> food proximity)
- "My children have nothing to do after school. We need a park." (social/culture need for children)
- "I studied medicine but there's no clinic. Build one so I can practice." (purpose need)
- "Three families share one well. We need a water tower." (infrastructure scaling)

### 6.3 Petition Resolution

Petitions accumulate signatures (more citizens with the same need = higher priority). The council reviews them during meetings. The player sees petitions ranked by urgency and can approve, defer, or dismiss them. Deferred petitions grow in urgency over time.

---

## 7. Building Variation

Not all buildings of the same type look identical. Each building gets a visual variant:

- **Houses** — 3-4 visual variants (color, roof style, garden).
- **Shops** — Signage and facade differ by business type.
- **Public buildings** — Unique per type but with wear/upgrade states.

When a building is placed, a variant is randomly selected from the available sprites for that type. Over time, buildings can be upgraded (visual change + stat improvement).

---

## 8. Events & Crises

### 8.1 Organic Events

These emerge from simulation state:
- **Wedding** — Two citizens with high relationship get married. Celebration at church/town hall. Boosts town morale.
- **Birth** — Married couple has a child. Increases housing pressure.
- **Death** — Elder or sick citizen dies. Emotional impact on family, frees housing.
- **Business opening** — Entrepreneur citizen opens a shop. New economic node.
- **Graduation** — Student completes education. Unlocks professional role.

### 8.2 Random Crises

- **Fire** — Destroys a building if no fire station. Rebuilding costs money.
- **Epidemic** — Spreads if no clinic/hospital. Citizens get sick, can't work.
- **Drought** — Farms produce less. Food prices spike.
- **Crime wave** — Theft, unrest. Worse without constabulary.
- **Economic downturn** — Trade income drops. Businesses may close.

### 8.3 Festivals & Culture

The council or citizens can petition for events:
- **Harvest Festival** — Boosts morale, costs money, requires festival grounds.
- **Market Fair** — Temporary trade boost, attracts visitors.
- **Town Anniversary** — Celebration, cultural boost.

---

## 9. Win/Loss Conditions

There's no hard win state — it's a sandbox. But there are milestones and failure states.

### Milestones
- First 10 residents
- First business opened by a citizen
- First child born in town
- First university graduate
- Population 50, 100, 200
- Self-sustaining economy (treasury grows without player intervention)
- Cultural landmark (museum/theater attracts tourism)

### Failure Triggers
- Treasury hits zero with debt — town goes bankrupt, services shut down, citizens leave.
- Population drops below 5 — town is abandoned.
- Satisfaction stays below 20% for extended time — mass exodus.

---

## 10. Sprite Requirements

**Style:** Isometric pixel art, 64x64 base tile, clean outlines, vibrant but not oversaturated palette. Think "Habbo Hotel meets SimCity 2000" — readable at small sizes, consistent lighting from top-left.

**Per-asset prompt template:**

```
Isometric pixel art sprite, 64x64 pixels, [SUBJECT], viewed from a 2:1 isometric angle
(30 degrees top-down), clean black outlines, flat shading with one highlight and one shadow tone,
transparent background, no anti-aliasing, consistent top-left lighting, game-ready
```

**Required sprite sheets:**

| Category | Assets Needed | Variants |
|----------|--------------|----------|
| **Terrain** | Grass tile, dirt tile, water tile, stone path | 2-3 each |
| **Town Hall** | Large civic building, flag, clock tower | 1 (unique) |
| **Houses** | Small cottage, medium house, apartment, manor | 3-4 color variants each |
| **Commercial** | Store, bakery, butcher, market stall, tavern, inn, blacksmith, tailor, bookstore | 1-2 each |
| **Civic** | School, library, university, clinic, hospital, church, courthouse, fire station, constabulary, jail | 1 each |
| **Infrastructure** | Well, water tower, granary, warehouse, trading post, bridge | 1 each |
| **Cultural** | Park (trees + bench), theater, museum, festival grounds | 1 each |
| **Industrial** | Farm, lumber mill, quarry, windmill | 1 each |
| **People** | Walking sprites (4 directions), idle, working | Male/female x 4 skin tones x 3 age groups |
| **UI** | Coin icon, heart, book, shield, food, hammer | 1 each |

---

## 11. LLM Integration Points

The LLM is called at these moments (never in the hot tick path):

1. **Mayor/Council generation** — Character creation with personality, bio, governing style.
2. **Council meetings** — Given town state, generate debate and proposal.
3. **New resident generation** — Given housing type, generate person/family.
4. **Citizen petitions** — Given unmet needs, generate petition text.
5. **Event narration** — Describe weddings, crises, festivals in character.
6. **Dialogue** — When citizens interact, generate in-character conversation.

All LLM calls include the building catalog so the model picks from valid options. All calls have template fallbacks.

---

## 12. Comparison: Old vs New

| Old smrti-town | New smrti-town |
|---------------|---------------|
| Pre-built world (Millbrook) | Empty field, player places Town Hall |
| Hardcoded 6 agents | LLM generates everyone |
| Primitive drawn shapes | Real isometric pixel art sprites |
| Flat economy (wallet + prices) | Full economic simulation (treasury, taxes, trade, entrepreneurship) |
| Rule-based decisions only | Council AI debates and proposes via LLM |
| Random population | Immigration driven by pull factors |
| Static skills | Skill development -> profession pipeline |
| Petitions detect keywords | Petitions emerge from unmet needs in the simulation |
| Buildings are boxes | Varied sprites per building type |
