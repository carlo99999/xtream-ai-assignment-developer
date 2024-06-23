# SQLAlchemy ORM Models for Diamond Price Prediction

This documentation provides an overview and explanation of the SQLAlchemy ORM models used in the diamond price prediction application. These models define the database schema and relationships between different entities.

## Database Configuration

First, the database URL is defined and the engine is created using SQLAlchemy. The `Base` class is also defined using SQLAlchemy's `declarative_base` function.

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey

SQLALCHEMY_DATABASE_URL = "sqlite:///./diamonds.db"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})

Base = declarative_base()
```

### Explanation
- **`create_engine`**: Creates the database engine.
- **`declarative_base`**: Returns a base class for the declarative model.

## Models

The following models are defined to represent the entities in the database.

### SavedDatas Model

The `SavedDatas` model stores information about the uploaded datasets.

```python
class SavedDatas(Base):
    __tablename__ = "SavedDatas"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String)
    model_id = Column(String)
```

### Explanation
- **`id`**: Primary key.
- **`file_name`**: Name of the uploaded file.
- **`model_id`**: Identifier for the model associated with the data.

### ToPredict Model

The `ToPredict` model stores information about the data that needs to be predicted.

```python
class ToPredict(Base):
    __tablename__ = "ToPredict"
    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(String)
    carat = Column(Float)
    cut = Column(String)
    color = Column(String)
    clarity = Column(String)
    depth = Column(Float)
    table = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    
    predictions = relationship("Predicted", back_populates="to_predict")
```

### Explanation
- **`id`**: Primary key.
- **`model_id`**: Identifier for the model used for prediction.
- **`carat`, `cut`, `color`, `clarity`, `depth`, `table`, `x`, `y`, `z`**: Attributes of the diamond.
- **`predictions`**: Relationship to the `Predicted` model.

### Predicted Model

The `Predicted` model stores the predictions made by the model.

```python
class Predicted(Base):
    __tablename__ = "Predicted"
    id = Column(Integer, primary_key=True, index=True)
    toPredictId = Column(Integer, ForeignKey("ToPredict.id"))
    model_id = Column(String)
    price = Column(Float)
    
    to_predict = relationship("ToPredict", back_populates="predictions")
```

### Explanation
- **`id`**: Primary key.
- **`toPredictId`**: Foreign key referencing `ToPredict`.
- **`model_id`**: Identifier for the model used for prediction.
- **`price`**: Predicted price of the diamond.
- **`to_predict`**: Relationship to the `ToPredict` model.

### Diamond Model

The `Diamond` model stores information about individual diamonds.

```python
class Diamond(Base):
    __tablename__ = "Diamonds"
    id = Column(Integer, primary_key=True, index=True)
    carat = Column(Float)
    cut = Column(String)
    color = Column(String)
    clarity = Column(String)
    
    similar_diamonds = relationship("SimilarDiamond", back_populates="diamond")
```

### Explanation
- **`id`**: Primary key.
- **`carat`, `cut`, `color`, `clarity`**: Attributes of the diamond.
- **`similar_diamonds`**: Relationship to the `SimilarDiamond` model.

### SimilarDiamond Model

The `SimilarDiamond` model stores information about diamonds that are similar to a specified diamond.

```python
class SimilarDiamond(Base):
    __tablename__ = "SimilarDiamonds"
    id = Column(Integer, primary_key=True, index=True)
    diamondId = Column(Integer, ForeignKey("Diamonds.id"))
    carat = Column(Float)
    cut = Column(String)
    color = Column(String)
    clarity = Column(String)
    price = Column(Float)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    table = Column(Float)
    depth = Column(Float)
    
    diamond = relationship("Diamond", back_populates="similar_diamonds")
```

### Explanation
- **`id`**: Primary key.
- **`diamondId`**: Foreign key referencing `Diamond`.
- **`carat`, `cut`, `color`, `clarity`, `price`, `x`, `y`, `z`, `table`, `depth`**: Attributes of the similar diamond.
- **`diamond`**: Relationship to the `Diamond` model.

## Summary

This documentation provides a detailed overview of the SQLAlchemy ORM models used in the diamond price prediction application. Each model represents a table in the SQLite database, defining the schema and relationships between different entities. This setup allows for efficient storage and retrieval of data related to diamond predictions.