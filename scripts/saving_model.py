import sklearn
import numpy
import json

#Un modèle est sauvé au format JSON
#chaque json est composé de trois clés au minimum
#la clé model ==> nom de la classe du modele permettant de le recharger plus tard grâce à new_model = eval(dictionnary["model"])(**dictionnary["params])
#la clé params permmetant de stocker les paramètres au sein d'un dictionnaire
#la clé attributes permettant de stocker les attributs fittés lors du training


def type_to_string(python_object):

    return str(type(python_object)).replace('<class ', '').replace('>', '').replace("'", "")

def make_it_json(bunch):

    python_type = type_to_string(bunch)

    if isinstance(bunch, numpy.float64):
        new_bunch = float(bunch)

    elif isinstance(bunch, numpy.int64):
        new_bunch = int(bunch)

    elif isinstance(bunch, numpy.ndarray):
        new_bunch = bunch.tolist()

    else:
        new_bunch = bunch

    return list((python_type, new_bunch))


def serialization(model):

    objects_list = (sklearn.tree._classes.DecisionTreeClassifier, sklearn.tree._classes.DecisionTreeRegressor)
    objects_builtin = (sklearn.tree._tree.Tree,)

    #Initialisation
    dico_serialization = dict()
    dico_serialization['model'] = dict()
    dico_serialization['params'] = dict()
    dico_serialization['attributes'] = dict()

    dico_serialization['model'] = type_to_string(model)

    if '__dict__' in dir(model):

        dico_serialization['params'] = dict(model.get_params())
        
        attributes = [element for element in model.__dict__ if element not in model.get_params()]
        
        for attribute, value in model.__dict__.items():
            #print(f"{attribute}/{value}")
            if attribute in dico_serialization['params']:
                pass
        
            elif isinstance(value, objects_list):
                dico_serialization['attributes'][attribute] = serialization(value)

            elif isinstance(value, list) and isinstance(value[0], objects_list):
                dico_serialization['attributes'][attribute] = [serialization(estimateur) for estimateur in value]
            
            elif isinstance(value, objects_builtin):

                if isinstance(value, sklearn.tree._tree.Tree):
                    n_features  = dico_serialization['attributes']['n_features_'][1]
                    
                    if isinstance(model, sklearn.tree._classes.DecisionTreeClassifier):
                        n_classes = dico_serialization['attributes']['n_classes_'][1]

                    else:
                        n_classes = 1

                    n_outputs = dico_serialization['attributes']['n_outputs_'][1]
                    
                    dico_serialization['attributes'][attribute] = serialize_tree(value, n_features, n_classes, n_outputs)
            else:
                
                dico_serialization['attributes'][attribute] = make_it_json(value)

    #print(dico_serialization)
    return dico_serialization


def serialize_tree(tree, n_features, n_classes, n_outputs):

    serialized_tree = dict()
    serialized_tree['model'] = "sklearn.tree._tree.Tree"
    serialized_tree['attributes'] = tree.__getstate__()

    dtypes = serialized_tree['attributes']['nodes'].dtype #TypeError: 'numpy.dtype' object is not iterable
    tree_dtypes = []
    for i in range(0, len(dtypes)):
        tree_dtypes.append(dtypes[i].str)
   
    serialized_tree['attributes']['nodes_dtype'] = tree_dtypes
    serialized_tree['attributes']['nodes'] = serialized_tree['attributes']['nodes'].tolist()
    serialized_tree['attributes']['values'] = serialized_tree['attributes']['values'].tolist()
    serialized_tree['attributes']['n_features'] = n_features
    serialized_tree['attributes']['n_classes'] = n_classes
    serialized_tree['attributes']['n_outputs'] = n_outputs

    return serialized_tree


def deserialize_tree(tree_dict):

    tree_dict['attributes']['nodes'] = [tuple(lst) for lst in tree_dict['attributes']['nodes']]

    names = ['left_child', 'right_child', 'feature', 'threshold', 'impurity', 'n_node_samples', 'weighted_n_node_samples']
    tree_dict['attributes']['nodes'] = numpy.array(tree_dict['attributes']['nodes'], dtype=numpy.dtype({'names': names, 'formats': tree_dict['attributes']['nodes_dtype']}))
    tree_dict['attributes']['values'] = numpy.array(tree_dict['attributes']['values'])

    tree = sklearn.tree._tree.Tree(tree_dict['attributes']['n_features'], numpy.array([tree_dict['attributes']['n_classes']], dtype=numpy.intp), tree_dict['attributes']['n_outputs'])
    tree.__setstate__(tree_dict['attributes'])

    return tree

def de_serialization(dictionnary): 

    array_types = list(("numpy.ndarray",))

    builtin_list = list(("sklearn.tree._tree.Tree",))

    if dictionnary['model'] in builtin_list:

        if dictionnary['model'] == "sklearn.tree._tree.Tree":
            loaded_model = deserialize_tree(dictionnary)
        
    else:
        #@TODO Make this eval safe
        loaded_model = eval(dictionnary['model'])(**dictionnary['params'])

        for attribute, value in dictionnary['attributes'].items():

            if isinstance(value, dict):
                setattr(loaded_model, attribute, de_serialization(value)) 
                #eval(f"loaded_model.{decision_tree} = de_serialization(valeur)")
                
            elif isinstance(value, list) and isinstance(value[0], dict):
                setattr(loaded_model, attribute, [de_serialization(element) for element in value])   
            
            else:
                if value[0] in array_types:
                    setattr(loaded_model, attribute, numpy.array(value[1]))#, dtype=value[0]))

                else:
                    #@TODO Make this eval safe
                    #print(attribute)
                    #print(value[0])
                    #print(array_types)
                    if value[0] == "NoneType":
                        setattr(loaded_model, attribute, None)
                    
                    else:
                        setattr(loaded_model, attribute, eval(value[0])(value[1]))

    return loaded_model


def model_to_json(model, file_name):

    with open(file_name, "w") as file:
        file.write(json.dumps(serialization(model)))#, indent=4))

    print(f"Model successfully save at {file_name} !")


def model_from_json(file_name):
    
    with open(file_name, "r") as file:
        dictionnary = json.loads(file.read())

    print(f"Model sucessfully loaded from {file_name} !")
    return de_serialization(dictionnary)


"""
Problèmes:

- si le modèle est un modele built-in pas de généralisation possible il faut le reconstruire à partir
d'un code en dur
Exemples:
Tree
OK DUR

- setattr(loaded_model, attribut, de_serialization(valeur)) NE MARCHE PAS POUR UN OBJET 
Il faut trouver un moyen d'attribuer un objet dans un attribut d'un objet déjà créer de manière générique
A L'AIR DE MARCHER

- lors de la dé-sérialization certains objets doivent etre remis au bon format. Pas de relation 1-1
Exemples: 
une liste de string doit rester une liste pour sklearn
une liste de nombre en mode tableau doit etre remis en numpy.array pour sklearn
SAUVER EGALEMENT LE TYPE SUR LEQUEL ON DOIT RETOMBER LORS DE LA DESERIALIZATION
"""