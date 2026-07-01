create table students (
    id uuid primary key,
    name text not null,
    created_at timestamptz not null default now()
);

create table attempts (
    id uuid primary key,
    student_id uuid references students(id),
    problem_id uuid,
    problem_text text not null,
    overlay_id text not null,
    constraints_fired text[] not null default '{}',
    was_under_constrained boolean not null default false,
    was_contradiction boolean not null default false,
    correct boolean not null default true,
    created_at timestamptz not null default now()
);

create table attempt_law_nodes (
    attempt_id uuid not null references attempts(id) on delete cascade,
    law_node text not null,
    primary key (attempt_id, law_node)
);

create table problem_bank (
    id uuid primary key,
    problem_text text not null,
    domain text not null,
    unknown_symbol text not null,
    known_symbols text[] not null,
    law_nodes text[] not null,
    constraints_fired text[] not null default '{}',
    created_at timestamptz not null default now()
);

create view weak_law_nodes as
select
    student_id,
    law_node,
    count(*) as attempts,
    count(*) filter (where not correct) as mistakes
from attempts
join attempt_law_nodes on attempt_law_nodes.attempt_id = attempts.id
group by student_id, law_node;

create index idx_attempts_student_created_at on attempts(student_id, created_at desc);
create index idx_problem_bank_domain_unknown on problem_bank(domain, unknown_symbol);
