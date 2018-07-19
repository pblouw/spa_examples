import numpy as np 
import nengo
import nengo_spa as spa

import matplotlib.pyplot as plt


dim = 32

###########
# Model 1 #
###########

# illustrates the basics of the new spa system
with spa.Network() as model:
    stim = spa.Transcode('Hello', output_vocab=dim)
    state = spa.State(dim)

    nengo.Connection(stim.output, state.input)
    probe = nengo.Probe(state.output, synapse=0.01)


sim = nengo.Simulator(model)
sim.run(0.5)

# plots raw vector dimensions
plt.plot(sim.trange(), sim.data[probe])
plt.xlabel('time (s)')
plt.show()

# plots vocab similarity, with legend of vocab keys
plt.plot(sim.trange(), spa.similarity(sim.data[probe], state.vocab))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(state.vocab.keys())
plt.show()


###########
# Model 2 #
###########

# illustrates providing input via Transcode
with spa.Network() as model:
    stim = spa.Transcode('RED*CIRCLE+BLUE*SQUARE', output_vocab=dim)
    query = spa.Transcode(lambda t: 'CIRCLE' if t < 0.25 else 'SQUARE', output_vocab=dim)
    state = spa.State(dim)

    stim * ~query >> state

    probe = nengo.Probe(state.output, synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[probe], state.vocab))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(state.vocab.keys())
plt.show()


###########
# Model 3 #
###########

# illustrates the new symbol syntax
with spa.Network() as model:

    query = spa.Transcode(lambda t: 'CIRCLE' if t < 0.25 else 'SQUARE', output_vocab=dim)
    state = spa.State(dim)


    (spa.sym.BLUE * spa.sym.SQUARE + spa.sym.RED * spa.sym.CIRCLE) * ~query >> state
    # spa.sym('RED*CIRCLE+BLUE*SQUARE') * ~query >> state

    probe = nengo.Probe(state.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[probe], state.vocab))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(state.vocab.keys())
plt.show()


###########
# Model 4 #
###########

# illustrates action selection system

def start(t):
    if t < 0.05:
        return 'A'
    else:
        return '0'

with spa.Network() as model:
    state = spa.State(dim)
    spa_input = spa.Transcode(start, output_vocab=dim)

    spa_input >> state

    with spa.ActionSelection():
        spa.ifmax(spa.dot(state, spa.sym.A), spa.sym.B >> state)
        spa.ifmax(spa.dot(state, spa.sym.B), spa.sym.C >> state)
        spa.ifmax(spa.dot(state, spa.sym.C), spa.sym.D >> state)
        spa.ifmax(spa.dot(state, spa.sym.D), spa.sym.E >> state)
        spa.ifmax(spa.dot(state, spa.sym.E), spa.sym.A >> state)

    probe = nengo.Probe(state.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[probe], state.vocab))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(state.vocab.keys())
plt.show()


###########
# Model 5 #
###########

# using spa.Transcode for computation
with spa.Network() as model:
    scene = spa.sym('RED*CIRCLE + BLUE*SQUARE')

    unbind = spa.Transcode(lambda t, x: scene * ~x, input_vocab=dim, output_vocab=dim)
    stim = spa.Transcode(lambda t: 'CIRCLE' if t < 0.25 else 'SQUARE', output_vocab=dim)
    state = spa.State(vocab=dim)

    stim >> unbind
    unbind >> state

    probe = nengo.Probe(state.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[probe], state.vocab))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(state.vocab.keys())
plt.show()


###########
# Model 6 #
###########

# vocabularies in the new system
vocab = spa.Vocabulary(dim)
vocab.populate('ONE; TWO; THREE; FOUR')

print(list(vocab.keys()))

vocab.populate('''
        X = ONE * TWO;
        Y = TWO * THREE;
        Z = THREE * FOUR;
        TEST = Z * ~FOUR
        ''')

print(list(vocab.keys()))

# compare vocab items for similarity
print(vocab['TEST'].dot(vocab['ONE']))
print(vocab['TEST'].dot(vocab['THREE']))

# get underlying vectors for vocab item
print(vocab['Z'].v)

# get subset of vocab (creates a new vocab)
number_vocab = vocab.create_subset(['ONE', 'TWO', 'THREE', 'FOUR'])
print(list(number_vocab.keys()))

# parse vs __getitem__ for vocabs?
key = number_vocab['FOUR']
parse = number_vocab.parse('FOUR')

print(key.__dict__)
print(parse.__dict__)
print(type(key), type(parse), key is parse)  # because __eq__ is not defined


vocab = spa.Vocabulary(dim)
vocab.add('A', vocab.create_pointer())
vocab.populate('B; C')
vocab.parse('A + B + C')

print(list(vocab.keys()))

###########
# Model 7 #
###########

# cleanup memories illustrated

with spa.Network() as model:
    scene = spa.sym.RED * spa.sym.SQUARE + spa.sym.BLUE * spa.sym.CIRCLE    
    stim = spa.Transcode(lambda t: 'CIRCLE' if t < 0.25 else 'SQUARE', output_vocab=dim)
    unbind = spa.Transcode(lambda t, x: scene * ~x, input_vocab=dim, output_vocab=dim)

    am = spa.WTAAssocMem(0.4, model.vocabs[dim])

    stim >> unbind
    unbind >> am

    probe = nengo.Probe(am.output, synapse=0.01)

sim = nengo.Simulator(model)
sim.run(0.5)

plt.plot(sim.trange(), spa.similarity(sim.data[probe], model.vocabs[dim]))
plt.xlabel('time (s)')
plt.ylabel('similarity')
plt.legend(model.vocabs[dim].keys())
plt.show()



###########
# Model 8 #
###########

# reinterpreting between vocabularies
v1 = spa.Vocabulary(dim)
v1.populate('A; B')
v2 = spa.Vocabulary(dim)
v2.populate('A; B')

with spa.Network() as model:
    state_1 = spa.State(v1)
    state_2 = spa.State(v2)

    spa.reinterpret(state_2) >> state_1
    spa.sym.A >> state_2

    probe = nengo.Probe(state_1.output, synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(0.5)


plt.plot(sim.trange(), v1['A'].dot(sim.data[probe].T))
plt.plot(sim.trange(), v2['A'].dot(sim.data[probe].T))
plt.xlabel("Time [s]")
plt.ylabel("Similarity")
plt.legend(["v1['A']", "v2['A']"])
plt.show()

print(model.vocabs)




###########
# Model 9 #
###########

# translating between vocabularies
v1 = spa.Vocabulary(dim)
v1.populate('A; B')
v2 = spa.Vocabulary(dim)
v2.populate('A; B')

with spa.Network() as model:
    state_1 = spa.State(v1)
    state_2 = spa.State(v2)

    spa.translate(state_2, v1) >> state_1
    spa.sym.A >> state_2

    probe = nengo.Probe(state_1.output, synapse=0.01)


with nengo.Simulator(model) as sim:
    sim.run(0.5)


plt.plot(sim.trange(), v1['A'].dot(sim.data[probe].T))
plt.plot(sim.trange(), v2['A'].dot(sim.data[probe].T))
plt.xlabel("Time [s]")
plt.ylabel("Similarity")
plt.legend(["v1['A']", "v2['A']"])
plt.show()

print(model.vocabs)