import scipy.io




a = scipy.io.loadmat('./annotations/hand_corrected/training-a_StateAns/a0001_StateAns.mat')
print(a)
print(a['state_ans'][0] )   # primera fila del diccionario
print(a['state_ans'][0][0])   # primer elemento de la primera fila del diccionario es la posicion de la marca
print(a['state_ans'][0][1])   # segundo elemento de la primera fila del diccionario sistole, diastolers, S2 o S2

for i in range(len(a['state_ans'])):
    #print(a['state_ans'][i][0][0][0])
    #print(a['state_ans'][i][1][0][0][0])
    print(str(a['state_ans'][i][0][0][0]) + '   ' + a['state_ans'][i][1][0][0][0])