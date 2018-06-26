import numpy as np

def KS_isvalid(solution, object_volumes, KS_volume):
    return np.sum(solution*object_volumes)<=KS_volume

def KS_initialize_population(n_individuals, object_volumes, KS_volume):
    s =np.random.randint(1,n_individuals)
    sol = np.zeros((s,len(object_volumes)))
    i=0
    while i<s:
        solution = np.random.randint(2, size=len(object_volumes))
        if KS_isvalid(solution,object_volumes, KS_volume):
            sol[i] = solution
            i+=1
    return sol

def main():
    n_objects         = 20
    KS_volume         = 50
    n_individuals     = 10

    max_object_value  = 100
    max_object_volume = 50
    
    object_values  = np.random.randint(max_object_value-1, size=n_objects)+1
    object_volumes = np.random.randint(max_object_volume-1, size=n_objects)+1
    pop = KS_initialize_population(n_individuals, object_volumes, KS_volume)
    print (pop)

    pop = KS_initialize_population(n_individuals, object_volumes, KS_volume)

    n_invalid = np.sum([1 for i in pop if not KS_isvalid(i, object_volumes, KS_volume)])

    if n_invalid != 0:
        print ("INCORRECTO!! Hay", n_invalid, "soluciones inválidas")
    else:
        print ("CORRECTO!!")

if __name__ == "__main__":
    main()